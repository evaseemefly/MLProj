import pathlib
import shutil
from pathlib import Path


def copy_files_with_pathlib(source_dir, target_dir):
    """
    使用 pathlib 将源目录的所有子目录中的文件，全部拷贝到目标目录。
    如果遇到同名文件，会自动重命名以避免覆盖。

    :param source_dir: 源目录路径 (字符串或 Path 对象)
    :param target_dir: 目标目录路径 (字符串或 Path 对象)
    """
    # 1. 将输入路径转换为 Path 对象，这是 pathlib 的核心
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # 2. 检查源目录是否存在
    if not source_path.is_dir():
        print(f"错误：源目录 '{source_path}' 不存在或不是一个目录。")
        return

    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)

    # 3. 检查并创建目标目录
    # parents=True 相当于 mkdir -p，会创建所有必需的父目录
    # exist_ok=True 表示如果目录已存在，则不抛出错误
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"目标目录 '{target_path}' 已准备就绪。")
    print("-" * 30)

    copied_count = 0
    skipped_count = 0
    renamed_count = 0

    # 4. 递归遍历源目录中的所有文件
    # source_path.rglob('*') 会递归地查找所有匹配项（文件和目录）
    for source_file in source_path.rglob('*'):
        # 确保我们只处理文件，跳过目录
        if source_file.is_file():
            # 5. 构建目标文件路径，使用 '/' 操作符，非常直观
            target_file_path = target_path / source_file.name

            # 6. 【核心】处理同名文件
            if target_file_path.exists():
                # 使用 .stem (文件名) 和 .suffix (扩展名) 属性来构建新名字
                base = target_file_path.stem
                suffix = target_file_path.suffix
                i = 1
                # 循环直到找到一个不存在的新文件名
                while target_file_path.exists():
                    new_name = f"{base}_{i}{suffix}"
                    target_file_path = target_path / new_name
                    i += 1

                print(f"警告：文件 '{source_file.name}' 已存在。将重命名为 '{target_file_path.name}'")
                renamed_count += 1

            # 7. 执行拷贝操作
            try:
                # shutil.copy2 可以无缝接受 Path 对象
                shutil.copy2(source_file, target_file_path)
                # Path 对象在打印时会自动转换为适合当前操作系统的路径字符串
                print(f"已复制: {source_file} -> {target_file_path}")
                copied_count += 1
            except Exception as e:
                print(f"错误：复制文件 '{source_file}' 时失败: {e}")
                skipped_count += 1

    print("-" * 30)
    print("操作完成！")
    print(f"总计：成功复制 {copied_count} 个文件 (其中 {renamed_count} 个被重命名), 跳过 {skipped_count} 个。")


def batch_copy_files2dir(source_directory: str, target_directory: str):
    source_path: pathlib.Path = Path(source_directory)

    subdirs = [item for item in source_path.iterdir() if item.is_dir()]

    for subdir in subdirs:
        sub_name: str = subdir.name
        target_station_path = pathlib.Path(destination_directory) / sub_name
        copy_files_with_pathlib(str(subdir), str(target_station_path))
        pass


def batch_ws_files2dir(source: str, target_dir: str):
    source_path: pathlib.Path = Path(source)
    target_path: pathlib.Path = Path(target_dir)
    subdirs = [item for item in source_path.iterdir() if item.is_dir()]
    for subdir in subdirs:
        copied_count = 0
        skipped_count = 0
        renamed_count = 0
        sub_name: str = subdir.name
        target_station_path = pathlib.Path(destination_directory) / sub_name
        for source_file in subdir.rglob('WS????_DAT.*'):
            if source_file.is_file():
                target_file_parent_dir = target_path / sub_name
                if not target_file_parent_dir.exists():
                    target_file_parent_dir.mkdir(parents=True, exist_ok=True)
                # 5. 构建目标文件路径，使用 '/' 操作符，非常直观
                target_file_path = target_file_parent_dir / source_file.name

                # 6. 【核心】处理同名文件
                if target_file_path.exists():
                    # 使用 .stem (文件名) 和 .suffix (扩展名) 属性来构建新名字
                    base = target_file_path.stem
                    suffix = target_file_path.suffix
                    i = 1
                    # 循环直到找到一个不存在的新文件名
                    while target_file_path.exists():
                        new_name = f"{base}_{i}{suffix}"
                        target_file_path = target_path / new_name
                        i += 1

                    print(f"警告：文件 '{source_file.name}' 已存在。将重命名为 '{target_file_path.name}'")
                    renamed_count += 1

                # 7. 执行拷贝操作
                try:
                    # shutil.copy2 可以无缝接受 Path 对象
                    shutil.copy2(source_file, target_file_path)
                    # Path 对象在打印时会自动转换为适合当前操作系统的路径字符串
                    print(f"已复制: {source_file} -> {target_file_path}")
                    copied_count += 1
                except Exception as e:
                    print(f"错误：复制文件 '{source_file}' 时失败: {e}")
                    skipped_count += 1
        print(f"已复制: {source_file} -> {target_file_path}")
        copied_count += 1
    pass


# --- 使用示例 ---
if __name__ == "__main__":
    # 请修改为您自己的路径
    # pathlib 可以很好地处理不同操作系统的路径分隔符
    source_directory = r"E:\01DATA\ML\渤海\渤海"  # <-- 需要拷贝的源文件夹
    destination_directory = r"E:\01DATA\ML\STATION_MERGE"  # <-- 所有文件最终存放的文件夹
    filter_dir = r'E:\01DATA\ML\STATION_FILTER_WS'
    # step1: 将原始路径下的所有海洋站的实况数据按照海洋站名称存储至目标路径下
    # batch_copy_files2dir(source_directory, destination_directory)
    # step2: 读取目标路径下按照站名批量读取 WS0303_DAT.01111 文件
    batch_ws_files2dir(destination_directory, filter_dir)
    pass
