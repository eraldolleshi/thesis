def append_bytes(source_file, destination_file):
    try:
        with open(source_file, 'rb') as source:
            with open(destination_file, 'ab') as destination:
                # Read all bytes from the source file
                bytes_to_append = source.read()

                # Append the bytes to the destination file
                destination.write(bytes_to_append)

        print(f"Bytes from '{source_file}' successfully appended to '{destination_file}'.")
    except FileNotFoundError:
        print("One or both of the specified files not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
source_file_path = 'emg_3_0.004_-128_1_True.bin'
destination_file_path = 'emg_3_0.005_-128_1_True.bin'

append_bytes(source_file_path, destination_file_path)
