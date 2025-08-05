def get_wind_sort_key(file_path):
    """从文件名中提取 YYYYMMDDHH 部分作为排序键"""
    # 这是一个更健壮的写法，以防文件名结构略有不同
    try:
        # 文件名示例: GRAPES_2024010100_240h_UV.nc
        # 分割后: ['GRAPES', '2024010100', '240h', 'UV.nc']
        date_time_str = file_path.name.split('_')[1]
        return date_time_str
    except IndexError:
        # 如果文件名不符合预期格式，返回一个空字符串，使其排在最前或最后
        return ""
