#!/bin/bash
# Initialize counter starting from 2 (as the new filenames start with 000002.png)
counter=1
# Loop over all PNG files in sorted order
for file in $(ls mihoyo/*.png | sort -V); do
    # Generate new filename with zero-padded 6-digit format
    newname=$(printf "mihoyo/%06d.png" $counter)
    echo "Renaming '$file' to '$newname'"
    mv "$file" "$newname"
    # Increment the counter
    counter=$((counter + 1))
done
