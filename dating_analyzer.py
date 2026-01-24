#!/usr/bin/env python3
"""
Dating Message Analyzer
Analyzes your dating conversations to find what messages work best.
"""

import sqlite3
import re
import argparse
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from pathlib import Path


def get_db_path():
    return Path.home() / "Library" / "Messages" / "chat.db"


def extract_text_from_blob(blob):
    """Extract text from attributedBody blob (NSAttributedString format)."""
    if not blob:
        return None
    try:
        # The attributedBody is a serialized NSAttributedString
        # The actual text is usually after "NSString" marker and before control chars

        # Method 1: Look for text between specific markers
        # NSAttributedString stores text early in the blob
        text_start = blob.find(b'\x01+')
        if text_start != -1:
            # Find the length byte and extract text
            chunk = blob[text_start+2:text_start+500]
            # Find printable ASCII sequence
            match = re.search(b'^[\x20-\x7e\xc0-\xff]+', chunk)
            if match:
                text = match.group().decode('utf-8', errors='ignore').strip()
                if is_valid_message(text):
                    return text

        # Method 2: Look after streamtyped marker
        marker = b'streamtyped'
        idx = blob.find(marker)
        if idx != -1:
            chunk = blob[idx+len(marker):idx+600]
            # Skip past the header bytes and find readable text
            matches = re.findall(b'[\x20-\x7e]{3,200}', chunk)
            for m in matches:
                text = m.decode('utf-8', errors='ignore').strip()
                if is_valid_message(text):
                    return text

        # Method 3: Scan for longest readable sequence that looks like a message
        matches = re.findall(b'[\x20-\x7e]{5,300}', blob)
        valid_matches = []
        for m in matches:
            text = m.decode('utf-8', errors='ignore').strip()
            if is_valid_message(text):
                valid_matches.append(text)

        if valid_matches:
            # Return the longest valid match
            return max(valid_matches, key=len)

    except Exception as e:
        pass
    return None


def is_valid_message(text):
    """Check if extracted text looks like a real message vs garbage."""
    if not text or len(text) < 2:
        return False

    # Skip known garbage patterns
    garbage_patterns = [
        r'^NS[A-Z]',           # NSString, NSValue, etc.
        r'^streamtyped',
        r'^\+\[',              # Objective-C method signatures
        r'^@"',
        r'^[\da-f]{8}-[\da-f]{4}',  # UUIDs
        r'^\)at_',             # Weird artifacts
        r'^Mat_',
        r'^[\x00-\x1f]',       # Control characters
        r'^bplist',            # Binary plist header
        r'^typedstream',
    ]

    for pattern in garbage_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return False

    # Should have at least some letters
    if not re.search(r'[a-zA-Z]', text):
        return False

    # Ratio of alphanumeric to total should be reasonable
    alnum = sum(1 for c in text if c.isalnum() or c.isspace())
    if alnum / len(text) < 0.5:
        return False

    return True


def find_chat_id(db_path, phone):
    """Find chat ID for a phone number."""
    # Normalize phone number
    digits = re.sub(r'\D', '', phone)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT ROWID, chat_identifier
        FROM chat
        WHERE chat_identifier LIKE ?
    """, (f'%{digits[-10:]}%',))
    result = cur.fetchone()
    conn.close()
    return result[0] if result else None


def get_conversation(db_path, chat_id):
    """Get all messages from a conversation."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT m.date, m.is_from_me, m.text, m.attributedBody
        FROM message m
        JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
        WHERE cmj.chat_id = ?
        ORDER BY m.date
    """, (chat_id,))

    messages = []
    for row in cur.fetchall():
        ts = datetime.fromtimestamp(row[0]/1000000000 + 978307200)
        is_me = bool(row[1])
        text = row[2] or extract_text_from_blob(row[3])

        if text:
            messages.append({
                'timestamp': ts,
                'is_me': is_me,
                'text': text
            })

    conn.close()
    return messages


class DatingAnalyzer:
    """Analyze dating conversations to find successful patterns."""

    def __init__(self):
        self.conversations = {}  # phone -> messages
        self.my_messages = []
        self.successful_messages = []
        self.openers = []
        self.responses_to_my_messages = []

    def add_conversation(self, name, messages):
        """Add a conversation to analyze."""
        self.conversations[name] = messages

        # Extract my messages and their responses
        for i, msg in enumerate(messages):
            if msg['is_me']:
                self.my_messages.append(msg)

                # Check if this is an opener (first message or >1hr gap)
                is_opener = (i == 0 or
                    (messages[i-1]['is_me'] == False and
                     (msg['timestamp'] - messages[i-1]['timestamp']).seconds > 3600))

                if is_opener:
                    self.openers.append({
                        'text': msg['text'],
                        'conversation': name
                    })

                # Look at response (if any)
                if i + 1 < len(messages) and not messages[i+1]['is_me']:
                    response = messages[i+1]
                    response_time = (response['timestamp'] - msg['timestamp']).seconds
                    response_length = len(response['text'])

                    self.responses_to_my_messages.append({
                        'my_message': msg['text'],
                        'response': response['text'],
                        'response_time_seconds': response_time,
                        'response_length': response_length,
                        'conversation': name
                    })

    def analyze_success(self):
        """Identify successful messages based on response quality."""

        results = {
            'total_my_messages': len(self.my_messages),
            'messages_with_responses': len(self.responses_to_my_messages),
            'openers': [],
            'best_messages': [],
            'patterns': {},
            'style': {}
        }

        # Analyze openers
        results['openers'] = self.openers

        # Score messages by response quality
        scored_messages = []
        for item in self.responses_to_my_messages:
            score = 0

            # Quick response = good (under 5 min)
            if item['response_time_seconds'] < 300:
                score += 3
            elif item['response_time_seconds'] < 1800:
                score += 1

            # Long response = engaged
            if item['response_length'] > 50:
                score += 3
            elif item['response_length'] > 20:
                score += 1

            # Question in response = interested
            if '?' in item['response']:
                score += 2

            # Exclamation = enthusiastic
            if '!' in item['response']:
                score += 1

            # Haha/lol = you're funny
            if re.search(r'\b(haha|lol|lmao|😂)\b', item['response'].lower()):
                score += 2

            scored_messages.append({
                **item,
                'score': score
            })

        # Sort by score
        scored_messages.sort(key=lambda x: x['score'], reverse=True)
        results['best_messages'] = scored_messages[:20]

        # Analyze patterns in successful messages
        successful = [m for m in scored_messages if m['score'] >= 4]

        if successful:
            # What makes your successful messages different?
            successful_texts = [m['my_message'] for m in successful]
            all_texts = [m['my_message'] for m in scored_messages]

            results['patterns'] = {
                'avg_length_successful': sum(len(t.split()) for t in successful_texts) / len(successful_texts) if successful_texts else 0,
                'avg_length_all': sum(len(t.split()) for t in all_texts) / len(all_texts) if all_texts else 0,
                'questions_in_successful': sum(1 for t in successful_texts if '?' in t) / len(successful_texts) if successful_texts else 0,
                'questions_in_all': sum(1 for t in all_texts if '?' in t) / len(all_texts) if all_texts else 0,
            }

        # Overall style
        all_my_texts = [m['text'] for m in self.my_messages]
        if all_my_texts:
            results['style'] = {
                'avg_words': sum(len(t.split()) for t in all_my_texts) / len(all_my_texts),
                'uses_periods': sum(1 for t in all_my_texts if t.endswith('.')) / len(all_my_texts),
                'uses_questions': sum(1 for t in all_my_texts if '?' in t) / len(all_my_texts),
                'all_lowercase': sum(1 for t in all_my_texts if t == t.lower()) / len(all_my_texts),
            }

        return results

    def get_report(self):
        """Generate a human-readable report."""
        analysis = self.analyze_success()

        lines = []
        lines.append("=" * 60)
        lines.append("DATING MESSAGE ANALYSIS")
        lines.append("=" * 60)

        lines.append(f"\nAnalyzed {len(self.conversations)} conversations")
        lines.append(f"Your messages: {analysis['total_my_messages']}")
        lines.append(f"Messages that got replies: {analysis['messages_with_responses']}")

        # Style summary
        if analysis['style']:
            s = analysis['style']
            lines.append(f"\n--- YOUR TEXTING STYLE ---")
            lines.append(f"Average message: {s['avg_words']:.1f} words")
            lines.append(f"Use periods: {s['uses_periods']:.0%}")
            lines.append(f"Ask questions: {s['uses_questions']:.0%}")
            lines.append(f"All lowercase: {s['all_lowercase']:.0%}")

        # Openers
        lines.append(f"\n--- YOUR OPENERS ---")
        for opener in analysis['openers'][:10]:
            lines.append(f'  "{opener["text"][:60]}" ({opener["conversation"]})')

        # Best messages
        lines.append(f"\n--- MESSAGES THAT WORKED BEST ---")
        lines.append("(Scored by: quick reply, long reply, questions back, laughter)")
        for msg in analysis['best_messages'][:15]:
            lines.append(f"\n  You: \"{msg['my_message'][:70]}\"")
            lines.append(f"  Her: \"{msg['response'][:70]}\"")
            lines.append(f"  Score: {msg['score']} | Reply in {msg['response_time_seconds']//60}min")

        # Patterns
        if analysis['patterns']:
            p = analysis['patterns']
            lines.append(f"\n--- WHAT WORKS ---")
            if p['avg_length_successful'] > p['avg_length_all']:
                lines.append(f"• Your successful messages are LONGER ({p['avg_length_successful']:.1f} vs {p['avg_length_all']:.1f} words)")
            else:
                lines.append(f"• Your successful messages are SHORTER ({p['avg_length_successful']:.1f} vs {p['avg_length_all']:.1f} words)")

            if p['questions_in_successful'] > p['questions_in_all']:
                lines.append(f"• Questions work well for you ({p['questions_in_successful']:.0%} of good messages have ?)")

        return '\n'.join(lines)

    def suggest_message(self, context=""):
        """Suggest a message based on what's worked."""
        analysis = self.analyze_success()

        lines = []
        lines.append("Based on your successful messages, try something like:\n")

        # Find relevant successful messages
        best = analysis['best_messages'][:10]

        if context:
            # Try to find messages similar to context
            context_words = set(context.lower().split())
            relevant = [m for m in best if context_words & set(m['my_message'].lower().split())]
            if relevant:
                best = relevant

        for msg in best[:5]:
            lines.append(f'• "{msg["my_message"]}"')
            lines.append(f'  (got: "{msg["response"][:50]}...")\n')

        # Style tips
        lines.append("\nYour winning formula:")
        s = analysis['style']
        lines.append(f"• Keep it around {s['avg_words']:.0f} words")
        if s['uses_questions'] > 0.3:
            lines.append("• Ask a question")
        if s['all_lowercase'] > 0.5:
            lines.append("• Stay lowercase")
        if s['uses_periods'] < 0.3:
            lines.append("• Skip the period at the end")

        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Dating Message Analyzer')
    parser.add_argument('phones', nargs='*', help='Phone numbers to analyze')
    parser.add_argument('--suggest', '-s', metavar='CONTEXT', help='Suggest a message (optional context)')
    parser.add_argument('--export', '-e', action='store_true', help='Export best messages to file')

    args = parser.parse_args()

    db_path = get_db_path()
    if not db_path.exists():
        print(f"Error: iMessage database not found")
        return 1

    if not args.phones:
        print("Usage: python3 dating_analyzer.py '+1234567890' '+0987654321' ...")
        print("\nOptions:")
        print("  --suggest 'topic'  Get message suggestions")
        print("  --export           Save best messages to file")
        return 1

    analyzer = DatingAnalyzer()

    for phone in args.phones:
        chat_id = find_chat_id(db_path, phone)
        if chat_id:
            messages = get_conversation(db_path, chat_id)
            if messages:
                # Use last 4 digits as name
                name = phone[-4:]
                analyzer.add_conversation(name, messages)
                print(f"✓ Loaded {len(messages)} messages from {phone}")
            else:
                print(f"✗ No messages found for {phone}")
        else:
            print(f"✗ Chat not found for {phone}")

    if not analyzer.conversations:
        print("\nNo conversations loaded!")
        return 1

    print("\n")

    if args.suggest:
        print(analyzer.suggest_message(args.suggest))
    else:
        print(analyzer.get_report())

    if args.export:
        analysis = analyzer.analyze_success()
        with open('best_messages.txt', 'w') as f:
            f.write("BEST DATING MESSAGES\n\n")
            for msg in analysis['best_messages']:
                f.write(f"Me: {msg['my_message']}\n")
                f.write(f"Her: {msg['response']}\n")
                f.write(f"Score: {msg['score']}\n\n")
        print("\nExported to best_messages.txt")

    return 0


if __name__ == '__main__':
    exit(main())
