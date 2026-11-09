#include "web_dashboard.h"
#include "config.h"
#include "esp_event.h"
#include "esp_check.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_timer.h"
#include "esp_task_wdt.h"
#include "nvs_flash.h"
#include <atomic>
#include <cstdio>
#include <cstring>

namespace {
const char* TAG = "kws_web";
std::atomic<uint8_t> last_id{255};
std::atomic<float> last_confidence{0.0f};
std::atomic<int64_t> last_seen_ms{0};
std::atomic<uint8_t> top_pred_id{255};
std::atomic<float> top_pred_conf{0.0f};
std::atomic<int16_t> live_audio_peak{0};
const char* labels[] = {"hello_jarvis", "unknown"};

void wifi_event_handler(void*, esp_event_base_t event_base, int32_t event_id, void*) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "Wi-Fi disconnected; retrying");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        esp_netif_t* netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
        esp_netif_ip_info_t ip_info = {};
        if (netif && esp_netif_get_ip_info(netif, &ip_info) == ESP_OK) {
            ESP_LOGI(TAG, "Dashboard available at http://" IPSTR, IP2STR(&ip_info.ip));
        }
    }
}

constexpr char kIndexHtml[] = R"HTML(
<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESP32 Keyword Spotter</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#f8fafc;max-width:620px;margin:24px auto;padding:0 18px}
main{background:#1e293b;border-radius:18px;padding:26px;box-shadow:0 12px 36px #0008;border:1px solid #334155}
h1{font-size:1.45rem;margin:0 0 4px;font-weight:700}
#status-badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.8rem;font-weight:600;margin-bottom:12px;background:#334155;color:#94a3b8}
#command{font-size:2.5rem;color:#67e8f9;margin:12px 0 4px;font-weight:800;letter-spacing:-0.5px;min-height:3.2rem;display:flex;align-items:center}
.muted{color:#94a3b8;font-size:0.92rem}
.card{background:#0b1120;border-radius:14px;padding:16px;margin:20px 0;border:1px solid #1e293b}
.meter-box{margin:10px 0}
.vu-bar{width:100%;height:14px;background:#1e293b;border-radius:7px;overflow:hidden;position:relative;margin:8px 0}
.vu-fill{height:100%;width:0%;background:linear-gradient(90deg,#38bdf8,#34d399,#facc15,#f87171);transition:width 120ms ease-out;border-radius:7px}
meter{width:100%;height:18px;border-radius:6px}
.row{display:flex;justify-content:space-between;align-items:center;margin-top:8px;font-size:0.92rem}
.badge{background:#334155;padding:3px 10px;border-radius:6px;font-weight:600}
.pulse{animation:pulse-anim 1s infinite alternate}
@keyframes pulse-anim{from{opacity:0.6}to{opacity:1}}
</style></head>
<body><main>
<h1>ESP32 Jarvis Wake Spotter</h1>
<div id="status-badge" style="background:#064e3b;color:#34d399">LISTENING</div>
<div id="command">👂 Listening...</div>
<div id="confidence" class="muted">Say: "Hello Jarvis, development mode on"</div>
<div class="meter-box" style="margin-top:14px">
  <div class="row"><span class="muted">Detection Confidence</span><span id="conf-val" class="badge">0%</span></div>
  <meter id="meter" min="0" max="1" value="0"></meter>
</div>

<div class="card">
  <div style="font-weight:600;margin-bottom:6px">Live Microphone Telemetry</div>
  <div class="meter-box">
    <div class="row"><span>Signal Amplitude</span><span id="mic-peak" class="badge">0</span></div>
    <div class="vu-bar"><div id="vu-level" class="vu-fill"></div></div>
    <div class="muted" style="font-size:0.78rem">Noise floor: &lt;5000 | Speech: 15,000–32,000</div>
  </div>
  <div class="row" style="margin-top:12px">
    <span>Live Candidate:</span>
    <span id="top-pred" class="badge" style="color:#38bdf8">silence</span>
  </div>
  <div class="row">
    <span class="muted">Trigger Threshold:</span>
    <span id="gate-thresh" class="muted">65%</span>
  </div>
</div>
<div class="row muted" style="font-size:0.8rem">
  <span>Core 1 DSP @ 10 kHz (2.8s Window)</span>
  <span>Connected to Wi-Fi</span>
</div>
</main>
<script>
let active = true;
async function poll() {
    try {
        const res = await fetch('/api/status', { cache: 'no-store' });
        if (res.ok) {
            const s = await res.json();
            const cmd = document.querySelector('#command');
            const conf = document.querySelector('#confidence');
            const confVal = document.querySelector('#conf-val');
            const badge = document.querySelector('#status-badge');
            const topPred = document.querySelector('#top-pred');
            const peakEl = document.querySelector('#mic-peak');
            const vu = document.querySelector('#vu-level');

            // Update VU meter bar (0 to 32767 peak)
            const vuPct = Math.min(100, Math.round((s.peak / 32767) * 100));
            vu.style.width = vuPct + '%';
            peakEl.textContent = s.peak;

            topPred.textContent = s.top + (s.top_conf > 0 ? (' (' + (s.top_conf * 100).toFixed(1) + '%)') : '');
            document.querySelector('#gate-thresh').textContent = (s.threshold * 100).toFixed(0) + '%';
            document.querySelector('#meter').value = s.confidence;
            confVal.textContent = (s.confidence * 100).toFixed(1) + '%';

            if (s.confidence > 0 && s.id === 0) {
                // Wake word triggered!
                cmd.textContent = '⚡ JARVIS ACTIVATED';
                cmd.style.color = '#4ade80';
                conf.textContent = 'Development Mode ON (' + (s.confidence * 100).toFixed(1) + '%)';
                badge.textContent = 'WAKE WORD DETECTED';
                badge.style.background = '#065f46';
                badge.style.color = '#34d399';
            } else if (s.active) {
                // Mic is actively hearing sound
                if (s.top === 'hello_jarvis' && s.top_conf >= 0.40) {
                    cmd.textContent = '🎙️ Hearing: Jarvis...';
                    cmd.style.color = '#facc15';
                    conf.textContent = 'Evaluating wake phrase (' + (s.top_conf * 100).toFixed(1) + '%)';
                } else {
                    cmd.textContent = '🎙️ Hearing Voice...';
                    cmd.style.color = '#38bdf8';
                    conf.textContent = 'Evaluating audio stream (Peak: ' + s.peak + ')';
                }
                badge.textContent = 'VOICE ACTIVE';
                badge.style.background = '#1e3a8a';
                badge.style.color = '#93c5fd';
            } else {
                // Quiet / Listening
                cmd.textContent = '👂 Listening...';
                cmd.style.color = '#94a3b8';
                conf.textContent = 'Say: "Hello Jarvis, development mode on"';
                badge.textContent = 'LISTENING';
                badge.style.background = '#064e3b';
                badge.style.color = '#34d399';
            }
        }
    } catch(e) {
        document.querySelector('#command').textContent = 'Disconnected';
        document.querySelector('#status-badge').textContent = 'OFFLINE';
    } finally {
        if (active) setTimeout(poll, 300);
    }
}
poll();
window.addEventListener('beforeunload', () => { active = false; });
</script></body></html>
)HTML";

esp_err_t index_handler(httpd_req_t* request) {
    httpd_resp_set_type(request, "text/html; charset=utf-8");
    return httpd_resp_send(request, kIndexHtml, HTTPD_RESP_USE_STRLEN);
}

esp_err_t favicon_handler(httpd_req_t* request) {
    httpd_resp_set_status(request, "204 No Content");
    return httpd_resp_send(request, nullptr, 0);
}

esp_err_t status_handler(httpd_req_t* request) {
    const uint8_t id = last_id.load();
    const int64_t seen_ms = last_seen_ms.load();
    const int64_t now_ms = esp_timer_get_time() / 1000;
    // Show recognized command for 3 seconds, then return to Listening...
    const bool is_recent = (seen_ms > 0) && (now_ms - seen_ms < 3000);
    const char* label = (is_recent && id == 0) ? "HELLO JARVIS (DEV MODE ON)" : "Listening...";
    const float conf = (is_recent && id == 0) ? last_confidence.load() : 0.0f;

    const uint8_t top_id = top_pred_id.load();
    const float top_conf = top_pred_conf.load();
    const char* top_label = (top_id == 255 || top_conf <= 0.0f) ? "silence" : (top_id < 2 ? labels[top_id] : "unknown");
    const int16_t peak = live_audio_peak.load();

    char response[300];
    std::snprintf(response, sizeof(response),
                  "{\"command\":\"%s\",\"confidence\":%.4f,\"id\":%u,"
                  "\"top\":\"%s\",\"top_conf\":%.4f,\"peak\":%d,\"threshold\":%.2f,\"active\":%s}",
                  label, conf, (is_recent && id == 0) ? 0 : 255, top_label, top_conf, peak, KWS_CONFIDENCE_THRESHOLD,
                  (peak >= KWS_SPEECH_MIN_PEAK) ? "true" : "false");
    httpd_resp_set_type(request, "application/json");
    httpd_resp_set_hdr(request, "Cache-Control", "no-store");
    return httpd_resp_send(request, response, HTTPD_RESP_USE_STRLEN);
}
    httpd_resp_set_type(request, "application/json");
    httpd_resp_set_hdr(request, "Cache-Control", "no-store");
    return httpd_resp_send(request, response, HTTPD_RESP_USE_STRLEN);
}

}

void kws_web_record_command(uint8_t command_id, float confidence, int64_t now_ms) {
    last_id.store(command_id);
    last_confidence.store(confidence);
    last_seen_ms.store(now_ms);
}

void kws_web_record_diag(uint8_t top_id, float top_confidence, int16_t audio_peak, int64_t now_ms) {
    top_pred_id.store(top_id);
    top_pred_conf.store(top_confidence);
    live_audio_peak.store(audio_peak);
}

void kws_web_update_audio(const int16_t*, size_t) {}

esp_err_t kws_web_start() {
    esp_err_t nvs_result = nvs_flash_init();
    if (nvs_result == ESP_ERR_NVS_NO_FREE_PAGES || nvs_result == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_RETURN_ON_ERROR(nvs_flash_erase(), TAG, "erase nvs");
        ESP_RETURN_ON_ERROR(nvs_flash_init(), TAG, "init nvs");
    } else {
        ESP_RETURN_ON_ERROR(nvs_result, TAG, "init nvs");
    }
    ESP_RETURN_ON_ERROR(esp_netif_init(), TAG, "netif");
    ESP_RETURN_ON_ERROR(esp_event_loop_create_default(), TAG, "event loop");
    esp_netif_create_default_wifi_sta();
    ESP_RETURN_ON_ERROR(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, nullptr), TAG, "wifi events");
    ESP_RETURN_ON_ERROR(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, nullptr), TAG, "ip events");
    wifi_init_config_t wifi_config = WIFI_INIT_CONFIG_DEFAULT();
    ESP_RETURN_ON_ERROR(esp_wifi_init(&wifi_config), TAG, "wifi init");
    wifi_config_t station = {};
    std::strncpy(reinterpret_cast<char*>(station.sta.ssid), KWS_WIFI_SSID, sizeof(station.sta.ssid));
    std::strncpy(reinterpret_cast<char*>(station.sta.password), KWS_WIFI_PASSWORD, sizeof(station.sta.password));
    ESP_RETURN_ON_ERROR(esp_wifi_set_mode(WIFI_MODE_STA), TAG, "wifi mode");
    ESP_RETURN_ON_ERROR(esp_wifi_set_config(WIFI_IF_STA, &station), TAG, "wifi config");
    ESP_RETURN_ON_ERROR(esp_wifi_start(), TAG, "wifi start");

    httpd_handle_t server = nullptr;
    httpd_config_t server_config = HTTPD_DEFAULT_CONFIG();
    server_config.lru_purge_enable = true;
    server_config.max_open_sockets = 7;
    ESP_RETURN_ON_ERROR(httpd_start(&server, &server_config), TAG, "http start");
    httpd_uri_t index = {.uri = "/", .method = HTTP_GET, .handler = index_handler, .user_ctx = nullptr};
    httpd_uri_t favicon = {.uri = "/favicon.ico", .method = HTTP_GET, .handler = favicon_handler, .user_ctx = nullptr};
    httpd_uri_t status = {.uri = "/api/status", .method = HTTP_GET, .handler = status_handler, .user_ctx = nullptr};
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(server, &index), TAG, "index handler");
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(server, &favicon), TAG, "favicon handler");
    ESP_RETURN_ON_ERROR(httpd_register_uri_handler(server, &status), TAG, "status handler");
    ESP_LOGI(TAG, "Connecting to Wi-Fi '%s'; the dashboard URL will be printed after DHCP", KWS_WIFI_SSID);
    return ESP_OK;
}
