output_file = 'file_names.txt'

# Open the output file in write mode
with open(output_file, 'w') as f:
    for i in range(1001):  # Generate numbers from 0 to 1000
        # Format the number to match the 00000.jpg format
        file_name = f"{i:05d}.jpg"
        # Write the file name to the text file
        f.write(file_name + '\n')

print(f"File names written to {output_file}")