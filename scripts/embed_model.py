"""Convert a .tflite model to a C++ byte array for ESP-IDF."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = args.model.read_bytes()
    values = ", ".join(f"0x{byte:02x}" for byte in data)
    args.output.write_text("#include \"model_data.h\"\n\nalignas(16) const unsigned char g_keyword_spotter_model[] = {\n  " + values + "\n};\nconst unsigned int g_keyword_spotter_model_len = sizeof(g_keyword_spotter_model);\n", encoding="utf-8")

if __name__ == "__main__":
    main()
