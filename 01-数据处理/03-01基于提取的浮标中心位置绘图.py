import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker


def draw_static_map():
    # 1. 准备数据
    # 格式: 浮标名称: {'color': 颜色, 'marker': 形状, 'points': [(经度, 纬度), ...]}
    buoy_data = {
        'MF01004': {
            'color': '#FF0000',  # 红色
            'marker': 'o',  # 圆点
            'points': [
                (119.8966, 37.9552)  # 阶段 1
            ]
        },
        'MF01002': {
            'color': '#00AA00',  # 绿色
            'marker': '^',  # 三角
            'points': [
                (120.0833, 39.0333),  # 阶段 1
                (120.0833, 39.0350),  # 阶段 2
                (120.0850, 39.0333),  # 阶段 3
                (120.0850, 39.0350)  # 阶段 4
            ]
        },
        'MF01001': {
            'color': '#0000FF',  # 蓝色
            'marker': 's',  # 方块
            'points': [
                (120.5951, 39.5005),  # 阶段 1
                (120.6099, 39.5051),  # 阶段 2
                (120.6094, 39.5031)  # 阶段 3
            ]
        }
    }

    # 2. 创建画布和投影
    # figsize=(12, 10) 保证图片足够大
    fig = plt.figure(figsize=(12, 10), dpi=300)
    # 使用 PlateCarree 投影 (普通的经纬度投影)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 3. 设置地图显示范围 (模拟 8-9 级缩放)
    # 渤海湾大致范围: 经度 117.5-122.5, 纬度 37.5-40.5
    extent = [117.5, 122.0, 37.5, 40.5]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # 4. 添加地理要素 (岸线、陆地、海洋)
    # scale='10m' 表示使用 1:1000万 的高分辨率数据 (需要下载，首次运行可能较慢)
    # 如果报错，可以尝试改为 '50m'
    res = '10m'

    # 陆地颜色
    ax.add_feature(cfeature.LAND.with_scale(res), facecolor='lightgray')
    # 海洋颜色
    ax.add_feature(cfeature.OCEAN.with_scale(res), facecolor='#E0F0FF')
    # 岸线轮廓
    ax.add_feature(cfeature.COASTLINE.with_scale(res), linewidth=1.2, edgecolor='black')
    # 省界/国界 (可选)
    ax.add_feature(cfeature.BORDERS.with_scale(res), linestyle=':', linewidth=0.5)

    # 5. 绘制浮标点
    print("正在绘制浮标点...")
    for name, data in buoy_data.items():
        points = data['points']
        lons = [p[0] for p in points]
        lats = [p[1] for p in points]

        # 绘制散点
        # s=100 点的大小, zorder=10 保证点在地图最上层
        ax.scatter(lons, lats,
                   color=data['color'],
                   marker=data['marker'],
                   s=100,
                   label=name,
                   edgecolor='white',
                   linewidth=1.5,
                   transform=ccrs.PlateCarree(),
                   zorder=10)

        # 添加文字标注 (仅在每组数据的第一个点旁标注，防止重叠)
        # 偏移量 (+0.05, -0.05) 是为了让文字不遮挡点
        ax.text(lons[0] + 0.08, lats[0] - 0.05, name,
                transform=ccrs.PlateCarree(),
                fontsize=12, fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
                zorder=11)

    # 6. 添加经纬度网格线
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    # 设置网格标签格式
    gl.top_labels = False  # 上方不显示经度
    gl.right_labels = False  # 右侧不显示纬度
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'gray'}
    gl.ylabel_style = {'size': 10, 'color': 'gray'}

    # 7. 添加标题和图例
    plt.title('Center Positions of Moored Buoys in Bohai Bay', fontsize=16, pad=20)
    plt.legend(loc='lower right', title='Buoy ID', frameon=True, shadow=True)

    # 8. 保存与显示
    output_filename = 'bohai_buoys_map_cartopy.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"绘图完成！图片已保存为: {output_filename}")
    plt.show()


if __name__ == "__main__":
    try:
        draw_static_map()
    except ImportError as e:
        print("错误: 请确保已安装 cartopy 和 matplotlib。")
        print(f"详细错误: {e}")
    except Exception as e:
        print(f"运行时发生错误: {e}")