#!/bin/bash
# Fix uploads directory permissions

echo "Fixing uploads directory permissions..."

# Create directory if it doesn't exist
mkdir -p uploads

# Set proper permissions - allow all users to read/write
chmod 777 uploads

# Fix ownership if it's not correct
chown -R 1000:1000 uploads 2>/dev/null || echo "Running as non-root user - ownership change skipped"

echo "Permissions fixed! Directory: uploads/"
echo "Permissions:"
ls -ld uploads/

