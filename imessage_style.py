#!/usr/bin/env python3
"""
iMessage Style Analyzer & Writer
Analyzes your texting style and helps write messages that sound like you.
"""

import sqlite3
import re
import os
import argparse
from datetime import datetime
from collections import Counter
from pathlib import Path


def get_db_path():
    """Get the iMessage database path."""
    return Path.home() / "Library" / "Messages" / "chat.db"


def extract_text_from_attributed_body(blob):
    """Extract readable text from attributedBody blob."""
    if not blob:
        return None
    try:
        # Method 1: Look for streamtyped NSString content
        matches = re.findall(b'NSString.{1,50}?([\x20-\x7e]{2,})', blob)
        if matches:
            return matches[-1].decode('utf-8', errors='ignore')

        # Method 2: Find readable ASCII sequences
        text_match = re.search(b'[\x20-\x7e]{10,}', blob)
        if text_match:
            return text_match.group().decode('utf-8', errors='ignore').strip()

        # Method 3: Try to find text after common markers
        for marker in [b'NSMutableString', b'NSString', b'+', b'\x01']:
            idx = blob.find(marker)
            if idx != -1:
                chunk = blob[idx:idx+500]
                readable = re.findall(b'[\x20-\x7e\n]{5,}', chunk)
                if readable:
                    return readable[0].decode('utf-8', errors='ignore').strip()
    except Exception:
        pass
    return None


def get_all_chats(db_path):
    """List all available chats."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT c.ROWID, c.chat_identifier, c.display_name,
               COUNT(cmj.message_id) as msg_count
        FROM chat c
        LEFT JOIN chat_message_join cmj ON c.ROWID = cmj.chat_id
        GROUP BY c.ROWID
        ORDER BY MAX(cmj.message_id) DESC
        LIMIT 50
    """)

    chats = cursor.fetchall()
    conn.close()
    return chats


def get_messages(db_path, chat_id=None, only_me=False, limit=None):
    """Extract messages from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT m.date, m.is_from_me, m.text, m.attributedBody, h.id
        FROM message m
        LEFT JOIN handle h ON m.handle_id = h.ROWID
        {}
        ORDER BY m.date DESC
        {}
    """.format(
        "JOIN chat_message_join cmj ON m.ROWID = cmj.message_id WHERE cmj.chat_id = ?" if chat_id else "",
        f"LIMIT {limit}" if limit else ""
    )

    if chat_id:
        cursor.execute(query, (chat_id,))
    else:
        cursor.execute(query)

    messages = []
    for row in cursor.fetchall():
        ts = datetime.fromtimestamp(row[0]/1000000000 + 978307200)
        is_from_me = bool(row[1])
        text = row[2] or extract_text_from_attributed_body(row[3])
        contact = row[4] or "Unknown"

        if text and (not only_me or is_from_me):
            messages.append({
                'timestamp': ts,
                'is_from_me': is_from_me,
                'text': text,
                'contact': contact
            })

    conn.close()
    return messages


class StyleAnalyzer:
    """Analyze texting style from messages."""

    def __init__(self, messages):
        self.messages = [m for m in messages if m['is_from_me']]
        self.texts = [m['text'] for m in self.messages]
        self.analysis = None

    def analyze(self):
        """Perform full style analysis."""
        if not self.texts:
            return {"error": "No messages to analyze"}

        all_text = ' '.join(self.texts)
        words = re.findall(r'\b\w+\b', all_text.lower())

        self.analysis = {
            'total_messages': len(self.texts),
            'avg_length': sum(len(t) for t in self.texts) / len(self.texts),
            'avg_words': sum(len(t.split()) for t in self.texts) / len(self.texts),

            # Punctuation style
            'uses_periods': sum(1 for t in self.texts if t.endswith('.')) / len(self.texts),
            'uses_exclamation': sum(1 for t in self.texts if '!' in t) / len(self.texts),
            'uses_question': sum(1 for t in self.texts if '?' in t) / len(self.texts),
            'uses_ellipsis': sum(1 for t in self.texts if '...' in t) / len(self.texts),

            # Capitalization
            'starts_caps': sum(1 for t in self.texts if t and t[0].isupper()) / len(self.texts),
            'all_lowercase': sum(1 for t in self.texts if t == t.lower()) / len(self.texts),

            # Emoji usage
            'uses_emoji': sum(1 for t in self.texts if re.search(r'[\U0001F300-\U0001F9FF]', t)) / len(self.texts),

            # Common words (excluding very common ones)
            'top_words': Counter(w for w in words if len(w) > 3).most_common(20),

            # Common phrases (2-grams)
            'common_phrases': self._get_ngrams(2, 10),

            # Message starters
            'common_starters': Counter(t.split()[0].lower() for t in self.texts if t.split()).most_common(10),

            # Sample messages (for reference)
            'sample_messages': self.texts[:20]
        }

        return self.analysis

    def _get_ngrams(self, n, top_k):
        """Extract common n-grams."""
        ngrams = []
        for text in self.texts:
            words = text.lower().split()
            for i in range(len(words) - n + 1):
                ngrams.append(' '.join(words[i:i+n]))
        return Counter(ngrams).most_common(top_k)

    def get_style_summary(self):
        """Get a human-readable style summary."""
        if not self.analysis:
            self.analyze()

        a = self.analysis

        summary = []
        summary.append(f"Analyzed {a['total_messages']} of your messages\n")

        # Length style
        if a['avg_words'] < 5:
            summary.append("• You keep it SHORT - averaging {:.1f} words per message".format(a['avg_words']))
        elif a['avg_words'] < 15:
            summary.append("• Medium-length messages - averaging {:.1f} words".format(a['avg_words']))
        else:
            summary.append("• You write LONG messages - averaging {:.1f} words".format(a['avg_words']))

        # Punctuation
        if a['uses_periods'] < 0.2:
            summary.append("• You rarely use periods at the end")
        elif a['uses_periods'] > 0.7:
            summary.append("• You almost always end with proper punctuation")

        if a['uses_exclamation'] > 0.3:
            summary.append("• You use exclamation marks a lot! ({:.0%} of messages)".format(a['uses_exclamation']))

        if a['uses_ellipsis'] > 0.1:
            summary.append("• You like using ellipsis... ({:.0%} of messages)".format(a['uses_ellipsis']))

        # Capitalization
        if a['all_lowercase'] > 0.7:
            summary.append("• all lowercase vibes ({:.0%} of messages)".format(a['all_lowercase']))
        elif a['starts_caps'] > 0.8:
            summary.append("• You capitalize properly ({:.0%} start with caps)".format(a['starts_caps']))

        # Emoji
        if a['uses_emoji'] > 0.2:
            summary.append("• Emoji user! ({:.0%} of messages have emoji)".format(a['uses_emoji']))
        elif a['uses_emoji'] < 0.05:
            summary.append("• Not big on emojis ({:.0%} usage)".format(a['uses_emoji']))

        # Top words
        summary.append("\nYour most-used words:")
        for word, count in a['top_words'][:10]:
            summary.append(f"  • {word} ({count}x)")

        # Common starters
        summary.append("\nYou often start messages with:")
        for word, count in a['common_starters'][:5]:
            summary.append(f"  • \"{word}...\" ({count}x)")

        return '\n'.join(summary)

    def suggest_rewrite(self, message):
        """Suggest how to rewrite a message in your style."""
        if not self.analysis:
            self.analyze()

        a = self.analysis
        result = message

        # Apply lowercase if that's the style
        if a['all_lowercase'] > 0.7:
            result = result.lower()

        # Remove period if that's the style
        if a['uses_periods'] < 0.2 and result.endswith('.'):
            result = result[:-1]

        # Add ellipsis if common
        if a['uses_ellipsis'] > 0.2 and len(result) > 20 and not result.endswith(('!', '?', '...')):
            result = result.rstrip('.') + '...'

        return result


def export_chat(db_path, chat_id, output_file=None):
    """Export a chat to readable format."""
    messages = get_messages(db_path, chat_id)
    messages.reverse()  # Chronological order

    lines = []
    for m in messages:
        sender = "Me" if m['is_from_me'] else m['contact']
        ts = m['timestamp'].strftime('%Y-%m-%d %H:%M')
        lines.append(f"{ts} | {sender}: {m['text']}")

    output = '\n'.join(lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
        print(f"Exported to {output_file}")
    else:
        print(output)


def main():
    parser = argparse.ArgumentParser(description='iMessage Style Analyzer & Writer')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List chats
    list_parser = subparsers.add_parser('list', help='List all chats')

    # Search chats
    search_parser = subparsers.add_parser('search', help='Search for a chat')
    search_parser.add_argument('query', help='Phone number or name to search')

    # Export chat
    export_parser = subparsers.add_parser('export', help='Export a chat')
    export_parser.add_argument('chat_id', type=int, help='Chat ID to export')
    export_parser.add_argument('-o', '--output', help='Output file')
    export_parser.add_argument('-n', '--limit', type=int, help='Limit messages')

    # Analyze style
    analyze_parser = subparsers.add_parser('analyze', help='Analyze your texting style')
    analyze_parser.add_argument('--chat', type=int, help='Specific chat ID (default: all)')
    analyze_parser.add_argument('--limit', type=int, default=1000, help='Messages to analyze')

    # Rewrite message
    rewrite_parser = subparsers.add_parser('rewrite', help='Rewrite a message in your style')
    rewrite_parser.add_argument('message', help='Message to rewrite')
    rewrite_parser.add_argument('--chat', type=int, help='Learn style from specific chat')

    # Style prompt (for use with AI)
    prompt_parser = subparsers.add_parser('prompt', help='Generate style prompt for AI')
    prompt_parser.add_argument('--chat', type=int, help='Specific chat ID')

    args = parser.parse_args()

    db_path = get_db_path()
    if not db_path.exists():
        print(f"Error: iMessage database not found at {db_path}")
        print("Make sure you're on macOS with Messages app.")
        return 1

    if args.command == 'list':
        chats = get_all_chats(db_path)
        print("ID\t| Contact\t\t\t| Messages")
        print("-" * 60)
        for chat in chats:
            name = chat[2] or chat[1] or "Unknown"
            print(f"{chat[0]}\t| {name[:30]:<30}\t| {chat[3]}")

    elif args.command == 'search':
        chats = get_all_chats(db_path)
        query = args.query.lower()
        for chat in chats:
            identifier = (chat[1] or '').lower()
            name = (chat[2] or '').lower()
            if query in identifier or query in name:
                display = chat[2] or chat[1]
                print(f"{chat[0]}\t| {display}")

    elif args.command == 'export':
        export_chat(db_path, args.chat_id, args.output)

    elif args.command == 'analyze':
        messages = get_messages(db_path, args.chat, only_me=True, limit=args.limit)
        analyzer = StyleAnalyzer(messages)
        print(analyzer.get_style_summary())

    elif args.command == 'rewrite':
        messages = get_messages(db_path, args.chat, only_me=True, limit=500)
        analyzer = StyleAnalyzer(messages)
        analyzer.analyze()

        original = args.message
        rewritten = analyzer.suggest_rewrite(original)

        print(f"Original:  {original}")
        print(f"Your style: {rewritten}")

    elif args.command == 'prompt':
        messages = get_messages(db_path, args.chat, only_me=True, limit=500)
        analyzer = StyleAnalyzer(messages)
        analysis = analyzer.analyze()

        # Generate a prompt describing the style
        print("=== STYLE PROMPT FOR AI ===\n")
        print("Write messages in this texting style:")
        print(f"- Average message length: {analysis['avg_words']:.0f} words")
        print(f"- Capitalization: {'lowercase' if analysis['all_lowercase'] > 0.7 else 'normal'}")
        print(f"- Ending punctuation: {'rarely' if analysis['uses_periods'] < 0.2 else 'usually'}")
        print(f"- Exclamation marks: {'often' if analysis['uses_exclamation'] > 0.3 else 'rarely'}")
        print(f"- Ellipsis: {'common' if analysis['uses_ellipsis'] > 0.1 else 'rare'}")
        print(f"- Emoji: {'frequent' if analysis['uses_emoji'] > 0.2 else 'minimal'}")
        print(f"\nCommon phrases: {', '.join(p[0] for p in analysis['common_phrases'][:5])}")
        print(f"\nExample messages from this person:")
        for msg in analysis['sample_messages'][:10]:
            print(f'  "{msg}"')

    else:
        parser.print_help()

    return 0


if __name__ == '__main__':
    exit(main())
