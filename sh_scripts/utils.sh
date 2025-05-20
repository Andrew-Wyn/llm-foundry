#!/bin/bash

# SOURCE_DIR="/leonardo_scratch/large/userexternal/lcolosi0/minerva/data/raw/redpajamas-head/1"
SOURCE_DIR="/leonardo_scratch/large/userexternal/lcolosi0/minerva/data/raw/redpajamas-middle/1"
cd "$SOURCE_DIR" || { echo "Failed to cd into $SOURCE_DIR"; exit 1; }

d1=1
d2=2
d3=3

type="json"

mkdir -p $d1 $d2 $d3

# Optional: create a log file
LOG_FILE="split_progress.log"
touch "$LOG_FILE"

# Count how many files we have and split plan
total=$(find . -maxdepth 1 -type f -name "*.$type" | wc -l)
split_size=$((total / 3))
remainder=$((total % 3))

echo "[INFO] Total JSON\JSONL files: $total"
echo "[INFO] Files per directory: $split_size (+$remainder in dir 3)"

# Get sorted file list
mapfile -t files < <(find . -maxdepth 1 -type f -name "*.$type" | sort)

counter=0
moved=0

for file in "${files[@]}"; do
    # Normalize path (strip leading ./)
    filename="${file#./}"

    # Skip if already moved
    if [[ -f "$d1/$filename" || -f "$d2/$filename" || -f "$d3/$filename" ]]; then
        continue
    fi

    # Determine target dir
    if (( counter < split_size )); then
        target="$d1"
    elif (( counter < 2 * split_size )); then
        target="$d2"
    else
        target="$d3"
    fi

    mv "$filename" "$target/" && moved=$((moved + 1)) || echo "[WARN] Failed to move $filename" >> "$LOG_FILE"

    counter=$((counter + 1))

    # Print progress every 1000 files
    if (( moved % 1000 == 0 )); then
        echo "[PROGRESS] Moved $moved / $total files at $(date)"
    fi
done

echo "[DONE] Total moved: $moved"



# #!/bin/bash

# LISTINGS_DIR="/leonardo/prod/data/ai/red_pajama/2.0/it/listings"
# PREFIX="/leonardo/prod/data/ai/red_pajama/2.0/it/documents"
# OUTPUT_BASE="/leonardo_scratch/large/userexternal/lcolosi0/minerva/data/raw"

# count=0
# copied=0
# skipped=0

# # Loop over each .txt file in the listings directory
# for listing_file in "$LISTINGS_DIR"/*.txt; do
#     echo "[INFO] Processing listing: $listing_file"

#     while IFS= read -r relative_path; do
#         full_path="${PREFIX}/${relative_path}.json"

#         # Determine section
#         if [[ "$relative_path" == *head ]]; then
#             SECTION="head"
#         elif [[ "$relative_path" == *middle ]]; then
#             SECTION="middle"
#         else
#             SECTION="unknown"
#         fi

#         filename="it_head_${count}.json"
#         output_path="${OUTPUT_BASE}/redpajamas-${SECTION}/${filename}"

#         # Check if file already exists
#         if [[ -f "$output_path" ]]; then
#             echo "[SKIP] $output_path already exists."
#             skipped=$((skipped + 1))
#         else
#             if cp "$full_path" "$output_path"; then
#                 echo "[COPY] $full_path -> $output_path"
#                 copied=$((copied + 1))
#             else
#                 echo "[ERROR] Failed to copy $full_path"
#             fi
#         fi

#         count=$((count + 1))

#         # Optional: progress summary every 100 files
#         if (( count % 100 == 0 )); then
#             echo "[PROGRESS] Processed: $count | Copied: $copied | Skipped: $skipped | $(date)"
#         fi

#     done < "$listing_file"

# done

# # Final summary
# echo "[DONE] Total processed: $count | Copied: $copied | Skipped: $skipped"
