#!/bin/bash

OUTPUT_FILE="${1:-project_dump.md}"
> "$OUTPUT_FILE"

echo "🔍 Начинаю сборку проекта в один файл..."
echo "📄 Выходной файл: $OUTPUT_FILE"

echo "# BULBA1-LIGHTNING PROJECT DUMP" >> "$OUTPUT_FILE"
echo "**Date:** $(date)" >> "$OUTPUT_FILE"
echo -e "---\n" >> "$OUTPUT_FILE"

# Максимальный размер файла для дампа (500 КБ)
MAX_SIZE_BYTES=512000

dump_file() {
    local file_path="$1"
    local file_size=$(wc -c < "$file_path" | tr -d ' ')
    
    # Защитная проверка: пропускаем слишком тяжелые файлы (датасеты, логи и т.д.)
    if [ "$file_size" -gt "$MAX_SIZE_BYTES" ]; then
        echo "⚠️ Пропущен (превышает 500 KB): $file_path ($file_size bytes)"
        return
    fi
    
    echo "================================================================" >> "$OUTPUT_FILE"
    echo "📁 FILE: $file_path ($file_size bytes)" >> "$OUTPUT_FILE"
    echo "================================================================" >> "$OUTPUT_FILE"
    
    local ext="${file_path##*.}"
    if [[ "$ext" == "py" ]]; then
        echo '```python' >> "$OUTPUT_FILE"
    elif [[ "$ext" == "rs" ]]; then
        echo '```rust' >> "$OUTPUT_FILE"
    elif [[ "$ext" == "toml" ]] || [[ "$ext" == "yaml" ]]; then
        echo '```yaml' >> "$OUTPUT_FILE"
    else
        echo '```text' >> "$OUTPUT_FILE"
    fi
    
    cat "$file_path" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo -e "\n\n" >> "$OUTPUT_FILE"
    echo "✅ Добавлен: $file_path"
}

find . -type f \
    -not -path "*/target/*" \
    -not -path "*/__pycache__/*" \
    -not -path "*/checkpoints/*" \
    -not -path "*/.git/*" \
    -not -path "*/venv/*" \
    -not -path "*/.venv/*" \
    -not -path "*/env/*" \
    -not -path "*/.env/*" \
    -not -path "*/node_modules/*" \
    -not -path "*/.pytest_cache/*" \
    -not -path "*/.mypy_cache/*" \
    -not -path "*/data_train/*" \
    -not -name "*.pyc" \
    -not -name "*.so" \
    -not -name "*.dylib" \
    -not -name "*.pt" \
    -not -name "*.jsonl" \
    -not -name "uv.lock" \
    -not -name "Cargo.lock" \
    -not -name "train_data.txt" \
    -not -name "dump_project.sh" \
    -not -name "$OUTPUT_FILE" \
    \( -name "*.py" -o -name "*.rs" -o -name "*.yaml" -o -name "*.toml" -o -name "*.md" -o -name "*.txt" \) | sort | while read -r file; do
    
    dump_file "$file"
done

FINAL_SIZE=$(wc -c < "$OUTPUT_FILE" | tr -d ' ')
echo "--------------------------------------------------"
echo "🎉 Готово! Сборка завершена."
echo "📦 Размер итогового дампа: $FINAL_SIZE bytes"
echo "📄 Файл сохранен как: $OUTPUT_FILE"