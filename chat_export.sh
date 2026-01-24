#!/bin/bash

# iMessage Chat Exporter
# Pulls conversations directly from the Messages database on Mac

DB_PATH="$HOME/Library/Messages/chat.db"

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo "Error: iMessage database not found at $DB_PATH"
    echo "Make sure you're running this on a Mac with Messages app."
    exit 1
fi

# Function to list recent conversations
list_chats() {
    echo "=== Recent Conversations ==="
    sqlite3 "$DB_PATH" "
        SELECT
            c.ROWID,
            c.chat_identifier,
            c.display_name,
            COUNT(cm.message_id) as msg_count
        FROM chat c
        LEFT JOIN chat_message_join cm ON c.ROWID = cm.chat_id
        GROUP BY c.ROWID
        ORDER BY MAX(cm.message_id) DESC
        LIMIT 20;
    " -separator ' | '
}

# Function to export a specific chat
export_chat() {
    local chat_id="$1"
    local limit="${2:-100}"

    echo "=== Chat Export ==="
    echo ""

    sqlite3 "$DB_PATH" "
        SELECT
            datetime(m.date/1000000000 + 978307200, 'unixepoch', 'localtime') as timestamp,
            CASE WHEN m.is_from_me = 1 THEN 'Me' ELSE COALESCE(h.id, 'Unknown') END as sender,
            CASE
                WHEN m.text IS NOT NULL THEN m.text
                WHEN m.cache_has_attachments = 1 THEN '[attachment]'
                ELSE '[empty]'
            END as message
        FROM message m
        LEFT JOIN handle h ON m.handle_id = h.ROWID
        INNER JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
        WHERE cmj.chat_id = $chat_id
        ORDER BY m.date ASC
        LIMIT $limit;
    " -separator ' | '
}

# Function to search for a contact
search_contact() {
    local search="$1"
    echo "=== Searching for: $search ==="
    sqlite3 "$DB_PATH" "
        SELECT DISTINCT
            c.ROWID,
            c.chat_identifier,
            c.display_name
        FROM chat c
        WHERE c.chat_identifier LIKE '%$search%'
           OR c.display_name LIKE '%$search%'
        LIMIT 10;
    " -separator ' | '
}

# Function to export chat to a clean text file
export_to_file() {
    local chat_id="$1"
    local output_file="${2:-chat_export.txt}"
    local limit="${3:-500}"

    sqlite3 "$DB_PATH" "
        SELECT
            datetime(m.date/1000000000 + 978307200, 'unixepoch', 'localtime') as timestamp,
            CASE WHEN m.is_from_me = 1 THEN 'Me' ELSE COALESCE(h.id, 'Unknown') END as sender,
            CASE
                WHEN m.text IS NOT NULL THEN replace(m.text, char(10), ' ')
                WHEN m.cache_has_attachments = 1 THEN '[attachment]'
                ELSE '[empty]'
            END as message
        FROM message m
        LEFT JOIN handle h ON m.handle_id = h.ROWID
        INNER JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
        WHERE cmj.chat_id = $chat_id
        ORDER BY m.date ASC
        LIMIT $limit;
    " -separator ' | ' > "$output_file"

    echo "Exported to $output_file"
}

# Main menu
case "$1" in
    list)
        list_chats
        ;;
    search)
        search_contact "$2"
        ;;
    export)
        export_chat "$2" "$3"
        ;;
    save)
        export_to_file "$2" "$3" "$4"
        ;;
    *)
        echo "iMessage Chat Exporter"
        echo ""
        echo "Usage:"
        echo "  $0 list                      - List recent conversations"
        echo "  $0 search <name/number>      - Search for a contact"
        echo "  $0 export <chat_id> [limit]  - Export chat to terminal"
        echo "  $0 save <chat_id> <file> [limit] - Save chat to file"
        echo ""
        echo "Example workflow:"
        echo "  1. $0 list                   # Find the chat_id (first column)"
        echo "  2. $0 export 42 50           # Preview last 50 messages from chat 42"
        echo "  3. $0 save 42 julia.txt 200  # Save 200 messages to julia.txt"
        ;;
esac
