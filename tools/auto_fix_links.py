import os
import re
import urllib.parse
from pathlib import Path

def get_all_md_files(base_dir):
    md_files = []
    for root, dirs, files in os.walk(base_dir):
        # Exclude hidden and specific dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'node_modules']]
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    return md_files

def build_file_map(base_dir):
    file_map = {}
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'node_modules']]
        for file in files:
            path = os.path.join(root, file)
            # Add exact name
            if path not in file_map.setdefault(file, []):
                file_map[file].append(path)
            # Add without prefix (like 01_)
            if re.match(r'^\d+_', file):
                no_prefix = re.sub(r'^\d+_', '', file)
                if path not in file_map.setdefault(no_prefix, []):
                    file_map[no_prefix].append(path)
            # Add url decoded
            decoded = urllib.parse.unquote(file)
            if decoded != file:
                if path not in file_map.setdefault(decoded, []):
                    file_map[decoded].append(path)
            
            # Lowercase matches
            if path not in file_map.setdefault(file.lower(), []):
                file_map[file.lower()].append(path)
    return file_map

def fix_links(base_dir):
    md_files = get_all_md_files(base_dir)
    file_map = build_file_map(base_dir)
    
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    
    fixed_count = 0
    
    for file_path in md_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        def replacer(match):
            nonlocal fixed_count
            text = match.group(1)
            url = match.group(2).strip()
            
            # Skip external and anchors
            if url.startswith(('http://', 'https://', 'ftp://', 'mailto:', 'tel:', '#')):
                return match.group(0)
            
            # Remove anchor for checking
            url_no_anchor = url.split('#')[0]
            anchor = '#' + url.split('#')[1] if '#' in url else ''
            
            # Skip empty url (just anchor)
            if not url_no_anchor:
                return match.group(0)
                
            decoded_path = urllib.parse.unquote(url_no_anchor)
            full_path = os.path.normpath(os.path.join(os.path.dirname(file_path), decoded_path))
            
            if os.path.exists(full_path):
                return match.group(0)
                
            # File doesn't exist, try to find it
            basename = os.path.basename(decoded_path)
            candidates = file_map.get(basename, [])
            
            if not candidates and basename.lower() in file_map:
                candidates = file_map.get(basename.lower(), [])
            
            # Try removing .md and looking for directory index or similar
            if not candidates and basename.endswith('.md'):
                pass
                
            if len(candidates) == 1:
                target_path = candidates[0]
                rel_path = os.path.relpath(target_path, os.path.dirname(file_path))
                # Fix windows slashes if any
                rel_path = rel_path.replace('\\', '/')
                
                # Make it start with ./ or ../
                if not rel_path.startswith('.'):
                    rel_path = './' + rel_path
                    
                new_url = urllib.parse.quote(rel_path) + anchor
                print(f"Fixed: {url} -> {new_url} in {file_path}")
                fixed_count += 1
                return f"[{text}]({new_url})"
            
            return match.group(0)
            
        new_content = re.sub(pattern, replacer, content)
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
    print(f"Total links fixed: {fixed_count}")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    fix_links(base_dir)
