#!/bin/bash

# Input file
input_file="cns_psychiatric_diseases_pmids_en.txt"

# Lines per chunk
lines_per_chunk=5000

# Total lines in the file
total_lines=$(wc -l < "$input_file")

# Calculate the number of chunks needed
num_chunks=$(( (total_lines + lines_per_chunk - 1) / lines_per_chunk ))

# Loop to split the file manually
for ((i=0; i<num_chunks; i++)); do
    start_line=$(( i * lines_per_chunk + 1 ))
    end_line=$(( start_line + lines_per_chunk - 1 ))
    chunk_file=$(printf "pmid_chunk_%03d.txt" "$i")
    
    # Extract the lines for this chunk
    sed -n "${start_line},${end_line}p" "$input_file" > "$chunk_file"
done
