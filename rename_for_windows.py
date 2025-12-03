import os
import sys

# Windows 文件名非法字符（必须移除）
WINDOWS_INVALID_CHARS = r'<>:"/\|?*'
# 需替换的特殊字符（重点：空格→空字符串，换行/回车→空字符串）
SPECIAL_CHARS = {
    ':': '-',
    '/': '-',
    '\\': '-',
    '|': '-',
    '?': '_',
    '*': '_',
    '"': '',
    '<': '[',
    '>': ']',
    '\n': '',  # 移除换行符（核心修复！）
    '\r': '',  # 移除回车符（核心修复！）
    '\t': '',  # 移除制表符
    # ' ': ''    # 移除所有可见空格（核心需求！）
}

def clean_filename(name):
    """清理文件名：移除非法字符、空格、换行符，符合 Windows 标准"""
    # 1. 替换/移除所有特殊字符（含空格、换行）
    for char, replace_char in SPECIAL_CHARS.items():
        if char in name:
            name = name.replace(char, replace_char)
    
    # 2. 移除 Windows 非法字符（双重保障）
    for char in WINDOWS_INVALID_CHARS:
        name = name.replace(char, '')
    
    # 3. 处理特殊情况（避免空文件名、末尾点等）
    name = name.strip()  # 去除首尾可能残留的空字符
    if len(name) == 0:
        return "untitled"  # 空名称替换为默认名
    if name.endswith('.'):
        name = name[:-1] + "_"  # 移除末尾的点（Windows 不允许）
    if name.startswith('.') and len(name) == 1:
        name = "dot_file"  # 单独一个点替换为默认名
    
    # 4. 限制文件名长度（Windows 上限 255 字符）
    max_len = 255
    if len(name) > max_len:
        if '.' in name:
            name_parts = name.rsplit('.', 1)
            name = name_parts[0][:max_len - len(name_parts[1]) - 1] + '.' + name_parts[1]
        else:
            name = name[:max_len]
    
    return name

def rename_target_files(root_dir):
    """仅处理 root_dir 下子文件夹中的文件（不修改文件夹名称）"""
    for root, dirs, files in os.walk(root_dir):
        # 仅处理 MARDM_SiT_XL_t2m 的直接子文件夹（如 63、64 等目录）
        if os.path.dirname(root) == root_dir:
            print(f"\n正在处理文件夹：{root}")
            for filename in files:
                old_path = os.path.join(root, filename)
                new_filename = clean_filename(filename)
                new_path = os.path.join(root, new_filename)
                
                if old_path != new_path:
                    try:
                        os.rename(old_path, new_path)
                        print(f"  ✅ {filename} → {new_filename}")
                    except Exception as e:
                        print(f"  ⚠️  失败 {filename}：{str(e)}")

if __name__ == "__main__":
    # 目标根目录（绝对路径）
    TARGET_ROOT = "/data/tiany/MARDM/generation/MARDM_SiT_XL_t2m"
    
    if not os.path.exists(TARGET_ROOT):
        print(f"❌ 错误：目录 {TARGET_ROOT} 不存在！")
        sys.exit(1)
    
    print(f"开始处理：{TARGET_ROOT} 下子文件夹中的文件")
    print("规则：移除所有空格/换行符、Windows 非法字符、限制长度\n")
    
    rename_target_files(TARGET_ROOT)
    
    print("\n✅ 处理完成！文件名称可正常通过 scp 传输到 Windows。")
