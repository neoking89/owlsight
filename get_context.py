import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SRC_DIR_PATH = os.path.join(PROJECT_ROOT, "src")

OUTPUT_FILE_NAME = "code_context.txt"
OUTPUT_FILE_PATH = os.path.join(PROJECT_ROOT, OUTPUT_FILE_NAME)


def write_file_content(output_file, file_path):
    """Reads content from file_path and writes it to output_file with a header."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        output_file.write(
            f"--- Content from: {os.path.relpath(file_path, PROJECT_ROOT)} ---\n\n"
        )
        output_file.write(content)
        output_file.write("\n\n")
        print(f"Successfully processed: {file_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


def main():
    """Main function to gather code content into the output file."""
    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as output_f:
        if os.path.isdir(SRC_DIR_PATH):
            for root, _, files in os.walk(SRC_DIR_PATH):
                for file_name in files:
                    if file_name.endswith(".py"):
                        file_path = os.path.join(root, file_name)
                        write_file_content(output_f, file_path)
        else:
            print(f"src directory not found at: {SRC_DIR_PATH}")

    print(f"\nAll content written to: {OUTPUT_FILE_PATH}")


if __name__ == "__main__":
    main()
