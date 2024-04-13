#%%
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll

from matplotlib.patches import PathPatch
from matplotlib.path import Path

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors

from mpl_toolkits.axes_grid1 import make_axes_locatable 
import plotly.figure_factory as ff

from scipy.ndimage import uniform_filter

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from geopy.distance import geodesic

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

import rasterio


#~~~을 본인의 bst_all.txt가 저장된 디렉터리로 바꾸기
base_dir =r"C:\Users\jjoo0\2024\strange\typhoon_JMA"
best_txt_path =rf"{base_dir}\bst_all.txt"

#지도 투영법 지정
projection = ccrs.LambertConformal(central_longitude=135, central_latitude=25,
                                   standard_parallels=(20, 40)) #지도에서 왜곡이 없는 지역

#태풍 타입 지정 dictionary
storm_type_dict = {
    2: "TC",
    3: "TS",
    4: "STS",
    5: "TY",
    6: "EX",
    7: "Just Enter",
    9: "TS over"
}

#끝 두자리만 보고 앞에 19 또는 20을 붙여서 연도 완성하는 함수
def convert_to_datetime(input_data):
    input_str = str(input_data)
    year_prefix = "19" if int(input_str[:2]) > 40 else "20"
    year = year_prefix + input_str[:2]
    month = input_str[2:4]
    day = input_str[4:6]
    hour = input_str[6:]
    return datetime.strptime(f"{year}{month}{day}{hour}", "%Y%m%d%H")

#best track 데이터 읽는 함수
def parse_typhoon_data(file_content):
    ##################################################################

    #66666이라는 숫자가 나오면 새로운 태풍의 정보가 시작된다는 뜻
    #66666이라는 숫자부터 다음 66666숫자까지 하나의 태풍 데이터
    #첫번째 줄에 태풍 id, 이름, 날짜변경선 건넘 여부, 마지막 베스트트렉 업데이트 날짜 수록
    #두번째 줄 이후에는 태풍 시간, 타입, 위경도, 중심기압, 최대풍속, 30kt, 50kt 풍속 반경 수록

    ##################################################################
    typhoon_blocks = file_content.split("66666")[1:]  # Skip the first empty split if any
    typhoons = {}
    for block in typhoon_blocks:
        lines = block.strip().split('\n')
        header = lines[0]


        # Extracting information from the header
        id = header.split()[0]
        tc_number_id = header.split()[2]
        dataline_flag = header.split()[4]

        if len(header.split()) == 8:
            storm_name = header.split()[6]
            last_version_date = header.split()[7]

        elif not header.split()[5].isdigit():
            storm_name = header.split()[5]
            last_version_date = header.split()[6]

        else:
            continue



        data = lines[1:]  # 첫 번째 줄은 이미 처리했으므로 제외하고 시작
        updated_data = []

        for line_data in data:
            split_data = line_data.split()
            while len(split_data) < 11:
                split_data.append(np.nan)  # 필요한 길이가 될 때까지 np.nan 추가
            updated_data.append(split_data)

        # 업데이트된 데이터로 DataFrame 생성
        data = pd.DataFrame(updated_data)

        storm_time = np.array([convert_to_datetime(i) for i in data.iloc[:, 0]])
        storm_type = np.array([storm_type_dict[int(i)] for i in data.iloc[:,2]])
        storm_lat  = np.array([int(i) for i in data.iloc[:,3]])/10  #원본데이터는 위경도가 10만큼 곱해져 있음
        storm_lon  = np.array([int(i) for i in data.iloc[:,4]])/10
        storm_mslp = np.array([int(i) for i in data.iloc[:,5]])
        storm_vmax = np.array([np.nan if pd.isna(i) else int(i) for i in data.iloc[:, 6]])/1.8532480 #kt -> m/s

        #50_dir은 50kt가 나타나는 가장 먼 지점의 방위
        #50_lrad은 50kt가 나타나는 가장 먼 지점과 중심과의 거리, srad는 가장 가까운 거리

        storm_v50_dir  = np.array([np.nan if pd.isna(i) or int(i) == 0 else int(i[0]) for i in data.iloc[:,7]])
        storm_v50_lrad = np.array([np.nan if pd.isna(i) or int(i) == 0 else int(i[1:])*1.8532480 for i in data.iloc[:,7]])  #mile -> km
        storm_v50_srad = np.array([np.nan if pd.isna(i) or int(i) == 0 else int(i)*1.8532480 for i in data.iloc[:,8]])      #mile -> km
        storm_v30_dir  = np.array([np.nan if pd.isna(i) or int(i) == 0 else int(i[0]) for i in data.iloc[:,9]])
        storm_v30_lrad = np.array([np.nan if pd.isna(i) or int(i) == 0 else int(i[1:])*1.8532480 for i in data.iloc[:,9]])  #mile -> km
        storm_v30_srad = np.array([np.nan if pd.isna(i) or int(i) == 0 else int(i)*1.8532480 for i in data.iloc[:,10]])     #mile -> km

        #각 정보들을 typhoons라는 딕셔너리에 저장
        typhoons[id] = {
            "name": storm_name,
            "tc_number_id": int(tc_number_id),
            "dataline_flag": int(dataline_flag),
            "last_version_date": last_version_date,
            "time": storm_time,
            "type": storm_type,
            "lat" : storm_lat,
            "lon" : storm_lon,
            "mslp": storm_mslp,
            "vmax": storm_vmax,
            "v50_dir": storm_v50_dir,
            "v50_lrad": storm_v50_lrad,
            "v50_srad": storm_v50_srad,
            "v30_dir": storm_v30_dir,
            "v30_lrad": storm_v30_lrad,
            "v30_srad": storm_v30_srad,
        }

    return typhoons

# 베스트트랙 txt를 읽고 tyhpoons 사전 생성
with open(best_txt_path, 'r') as file:
    file_content = file.read()

typhoons = parse_typhoon_data(file_content)


#위경도를 주었을 때 실제 거리를 구하는 함수
def haversine_distance(lat1, lon1, lat2, lon2):
    # 모든 각도를 라디안으로 변환
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # 위도와 경도 차이
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine 공식
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # 6,371 km는 지구의 반지름
    distance = 6371 * c
    return distance



def draw_radii(ax, lon, lat, bearing, distance_s, distance_l, edgecolor='red', linewidth = 1, zorder = 5):
    ##########

    # v50, v30 최대 풍속이 가장 멀리 나타나는 방위, 거리, 가장 짧은 거리를 입력으로 받고
    # 이때 얻게 되는 두 지점을 원으로 그림에 표시하는 함수
    
    ##########
    path_coords = []

    if np.isnan(bearing):
        return 
    
    distance_to_move = (distance_l - distance_s) / 2
    new_distance = (distance_l + distance_s) / 2
    new_center = geodesic(kilometers=distance_to_move).destination((lat, lon), bearing*45)


    for bearing_deg in np.linspace(0,360,361):
        destination = geodesic(kilometers=new_distance).destination((new_center.latitude, new_center.longitude), bearing_deg)
        x, y = projection.transform_point(destination.longitude, destination.latitude, ccrs.Geodetic())
        path_coords.append((x, y))


    path = Path(path_coords)
    patch = PathPatch(path,facecolor = 'none', edgecolor=edgecolor, linewidth=linewidth, alpha=1, zorder = zorder)
    ax.add_patch(patch)


#lat1, lat2간 방향을 지정
def calculate_bearing(lat1, lon1, lat2, lon2):
    # 위도, 경도를 라디안으로 변환
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    
    # 경도 차이
    dLon = lon2 - lon1
    
    # 방위 계산
    x = np.sin(dLon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dLon))
    initial_bearing = np.arctan2(x, y)
    
    # 방위를 도 단위로 변환하고 0~360 범위로 조정
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    
    return compass_bearing

#각을 입력 받으면 16방위로 변환
def direction_16(bearing):
    compass_brackets = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N"]
    bracket_size = 360 / 16
    index = int((bearing + bracket_size/2) // bracket_size)
    return compass_brackets[index % 16]


#x, y을 기반으로 선을 연결하고 z값을 이용해 색상 표현
def colorline(ax, x, y, z=None, cmap=plt.get_cmap('jet_r'), norm=mcolors.Normalize(vmin=900, vmax=1020), linewidth=2, alpha=1.0, zorder=5, transform = ccrs.PlateCarree()):
    # x, y는 선의 좌표, z는 색상에 사용될 값
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    
    # z 값을 정규화
    z = np.asarray(z)

    # 선분을 색상으로 구분하여 그리기
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha, zorder=zorder, transform = transform)
    
    ax.add_collection(lc)
    
    return lc


#JMA 카테고리 등급 표현 (0~5 등급이 TD(Tropical Depression)~VT(Violent Typhoon)에 해당함)
jma_category = [0, 17, 25, 33, 44, 54]

def find_jma_index(vmaxs, jma_category):
    # vmaxs 각 요소에 대한 인덱스 리스트 초기화
    index_list = []
    

    for vmax in vmaxs:
        # vmax가 위치하는 인덱스 찾기
        for i in range(len(jma_category) - 1):
            if jma_category[i] <= vmax < jma_category[i + 1]:
                index_list.append(i)
                break
        else:
            # vmax가 jma_category의 마지막 경계값 이상인 경우 5 부여
            if vmax >= jma_category[-1]:
                index_list.append(len(jma_category) - 1)
    
    return index_list


# gray_cmap = LinearSegmentedColormap.from_list('custom_gray', [(0.2,0.2,0.2),(0.8,0.8,0.8)], N=256)

# Open your GeoTIFF file
# file_path = r"~~~~~~~~~\SR_LR\SR_LR.tif"
# with rasterio.open(file_path) as src:
#     image = src.read(1)  # Read the first band
#     image = uniform_filter(image, size=(10, 10))
#     image = image[::5,::5]//10
#     bounds = src.bounds


#태풍 년도, 번호를 입력으로 받고 typhoon에서 해당하는 태풍을 반환
def get_storm_info():
    while True:
        storm_year = input("Storm year (YYYY format): ")
        if storm_year.lower() == "exit":
            print("Exiting program.")
            return None  
        if len(storm_year) != 4 or not storm_year.isdigit():
            print("Please enter a valid year in YYYY format.")
            continue
        
        year_suffix = storm_year[2:]
        valid_storms = {k: v for k, v in typhoons.items() if k.startswith(year_suffix)}
        
        if not valid_storms:
            print(f"No storms recorded for {storm_year}.")
            continue

        while True:
            storm_input_str = ''
            for key in sorted(valid_storms.keys()):
                storm_input_str += f"{key[-2:]}: {valid_storms[key]['name']}\n"
            storm_num = input("Choose a storm number, type 'year' to re-enter year, or type 'exit' to exit:\n" + storm_input_str)
            
            if storm_num.lower() == "exit":
                print("Exiting program.")
                return None
            if storm_num.lower() == "year":
                break  # Break the inner loop to return to the year input
            
            if len(storm_num) < 2:
                storm_num = '0' + storm_num
            
            storm_id = year_suffix + storm_num
            
            if storm_id in valid_storms:
                return valid_storms[storm_id]
            else:
                print("Invalid storm number. Please try again.")

#%%
#태풍 년도 입력 => 번호 입력(exit 입력 시 탈출)
storm = get_storm_info()
print("Selected Storm: ", storm['name'])

#TD, EX 제거 TD 마지막, EX 처음 것만 살리기
TS_mask = (storm['type']=='TS') | (storm['type']=='STS') | (storm['type']=='TY') | (storm['type']=='TS over') | (storm['type']=='Just Enter')
start_idx, end_idx = max(0, np.where(TS_mask==True)[0][0]-1), min(np.where(TS_mask==True)[0][-1]+1,len(TS_mask))

for key in storm:
    if type(storm[key]) == np.ndarray:
        if len(storm[key]) == len(TS_mask):
            storm[key] = storm[key][start_idx:end_idx+1]


for storm_time in storm['time']:

    extent = 'follow'

    if extent == "follow":
        cen_lon = storm['lon'][storm['time']==storm_time][0]
        cen_lat = storm['lat'][storm['time']==storm_time][0]
        
        #극에 가까워질수록 경도가 좁아지는 것 보완
        adjustment_factor = 1 / np.cos(np.radians(cen_lat))
        adjusted_lon_range = 10 * adjustment_factor
        extent = [cen_lon-adjusted_lon_range, cen_lon+adjusted_lon_range, cen_lat-10, cen_lat+10]

    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})
    ax.set_title(f"JMA Best Track\n{storm_time.strftime('%Y.%m.%d.%HUTC')}", loc = 'left', fontsize=40, fontweight = 'bold')

    if ~np.isnan(storm['vmax'][storm['time']==storm_time][0]) and storm['vmax'][storm['time']==storm_time][0] != 0:
        ax.set_title(f"{storm['name'].upper()}\n{storm['mslp'][storm['time']==storm_time][0]}hPa, {storm['vmax'][storm['time']==storm_time][0]:.1f}m/s"
                    , loc = 'right', fontsize=40, fontweight = 'bold')
    else:    
        ax.set_title(f"{storm['name'].upper()}\n{storm['mslp'][storm['time']==storm_time][0]}hPa"
                    , loc = 'right', fontsize=40, fontweight = 'bold')


    ax.set_extent(extent, crs=ccrs.PlateCarree())
    land_color = mcolors.to_rgba((127/255, 153/255, 127/255))
    ocean_color = mcolors.to_rgba((190/255, 210/255, 254/255))
    coastline_10m = cfeature.NaturalEarthFeature('physical', 'coastline', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='none', facecolor=land_color)
    ocean_10m = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', edgecolor='none', facecolor=ocean_color)

    # ax.imshow(image, origin='upper', transform=projection, cmap=gray_cmap, vmin=5, vmax=20, interpolation='nearest')
    ax.add_feature(ocean_10m)
    ax.add_feature(land_10m)
    ax.add_feature(coastline_10m)


    #5도 간격으로 위경도 표시
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, 
                    xlocs = np.arange(-180, 180,5), ylocs = np.arange(-90, 90, 5), 
                    rotate_labels = 0, xpadding = -10, ypadding = -10)
    gl.top_labels = False
    gl.xlabels = False
    gl.left_labels = False

    gl.xlabel_style = {'size': 18, 'color': 'gray'}
    gl.ylabel_style = {'size': 18, 'color': 'gray'}


    for lon, lat, bearing_50, distance_50s, distance_50l, bearing_30, distance_30s, distance_30l, s_time in zip(storm['lon'], storm['lat'], storm['v50_dir'], storm['v50_lrad'], storm['v50_srad'],
                                                                                                        storm['v30_dir'], storm['v30_lrad'], storm['v30_srad'], storm['time']):
        
        if s_time == storm_time:
            draw_radii(ax, lon, lat, bearing_50, distance_50s, distance_50l, edgecolor='red', linewidth = 6, zorder = 5)
            draw_radii(ax, lon, lat, bearing_30, distance_30s, distance_30l, edgecolor='yellow', linewidth = 4, zorder=5)


    vmaxs = storm['vmax'][storm['time']<=storm_time]
    times = storm['time'][storm['time']<=storm_time]
    lons = storm['lon'][storm['time']<=storm_time]
    lats = storm['lat'][storm['time']<=storm_time]

    jet_cmap = plt.get_cmap('turbo')
    colors = jet_cmap(np.linspace(0, 1, 6))
    jet_category = ListedColormap(colors)

    #최대 풍속이 안 나와있으면 그냥 선으로, 나와있으면 그에 상응하는 색의 선으로 표현
    if np.all(np.isnan(storm['vmax'])):
        ax.plot(lons, lats, color = (7/255,15/255,152/255), transform = ccrs.PlateCarree(), linewidth = 2)
    
    else:
        category_wind = find_jma_index(vmaxs, jma_category)
        cl = colorline(ax, lons, lats, category_wind, cmap=jet_category, norm=mcolors.Normalize(vmin=0, vmax=len(jma_category)-1), linewidth=2, alpha=1.0, zorder=5)
        cbar = fig.colorbar(cl, ax=ax, orientation="horizontal", pad=0.01)
        cbar.set_ticks(np.arange(5/12, 5, 5/6))
        cbar.set_ticklabels(['TD', 'TS', 'STS', 'TY', 'VST', 'VT']) 
        cbar.ax.tick_params(direction='in', length=0, labelcolor='white', pad=-40)

    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_color('white')
        label.set_fontsize(30)

        
    ax.scatter(lons[-1], lats[-1], s=300, color='black',marker = 'x' ,linewidths=5,transform = ccrs.PlateCarree(), zorder = 4)

    if len(lons) > 1:
        moving_dis = haversine_distance(lats[-1], lons[-1], lats[-2], lons[-2])
        moving_time = (times[-1] - times[-2]).total_seconds() / 3600
        speed = moving_dis / moving_time  # km/h 단위로 속도 계산
        bearing = calculate_bearing(lats[-2], lons[-2], lats[-1], lons[-1])
        direction = direction_16(bearing)
        ax.text(0.915, 0.025, f'{direction:>3} {speed:>4.1f} km/h',fontsize=20, fontweight='bold', color='black', ha='center', va='top', transform=ax.transAxes)


    ax.set_aspect(1, adjustable="datalim")

    folder_path = rf"{base_dir}\bst_img\{storm['time'][0].strftime('%Y')}_{storm['name']}"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(rf"{folder_path}\{storm_time.strftime('%Y.%m.%d.%HUTC')}.png", bbox_inches = 'tight')
    plt.close()


#전체 경로 그리기
extent = [105,165,0,45]

fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})
ax.set_title(f"JMA Best Track\n{storm['time'][1].strftime('%Y.%m.%d')} - {storm['time'][-1].strftime('%m.%d')}", loc = 'left', fontsize=40, fontweight = 'bold')


if np.all(np.isnan(storm['vmax'])):
    ax.set_title(f"{storm['name'].upper()}\n{np.nanmin(storm['mslp']):.0f}hPa", loc = 'right', fontsize=40, fontweight = 'bold')
else:
    ax.set_title(f"{storm['name'].upper()}\n{np.nanmin(storm['mslp']):.0f}hPa, {np.nanmax(storm['vmax']):.1f}m/s", loc = 'right', fontsize=40, fontweight = 'bold')


#배경 지정하기
ax.set_extent(extent, crs=ccrs.PlateCarree())
land_color = mcolors.to_rgba((127/255, 153/255, 127/255))
ocean_color = mcolors.to_rgba((190/255, 210/255, 254/255))
coastline_10m = cfeature.NaturalEarthFeature('physical', 'coastline', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='none', facecolor=land_color)
ocean_10m = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', edgecolor='none', facecolor=ocean_color)

# ax.imshow(image, origin='upper', transform=projection, cmap=gray_cmap, vmin=5, vmax=20, interpolation='nearest')
ax.add_feature(ocean_10m)
ax.add_feature(land_10m)
ax.add_feature(coastline_10m)


#5도 간격으로 위경도 표시
gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, 
                xlocs = np.arange(-180, 180,5), ylocs = np.arange(-90, 90, 5), 
                rotate_labels = 0, xpadding = -10, ypadding = -10)
gl.top_labels = False
gl.xlabels = False
gl.left_labels = False

gl.xlabel_style = {'size': 18, 'color': 'gray'}
gl.ylabel_style = {'size': 18, 'color': 'gray'}


#태풍 등급 색상 만들기
jet_cmap = plt.get_cmap('turbo')
colors = jet_cmap(np.linspace(0, 1, 6))
jet_category = ListedColormap(colors)


#최대 풍속이 안 나와있으면 그냥 선으로(옛날 태풍은 최대 풍속 기록 X), 나와있으면 그에 상응하는 색의 선으로 표현
if np.all(np.isnan(storm['vmax'])):
    ax.plot(storm['lon'], storm['lat'], color = (7/255,15/255,152/255), transform = ccrs.PlateCarree(), linewidth = 2)

else:
    category_wind = find_jma_index(vmaxs, jma_category)
    cl = colorline(ax, storm['lon'], storm['lat'], category_wind, cmap=jet_category, norm=mcolors.Normalize(vmin=0, vmax=len(jma_category)-1), linewidth=2, alpha=1.0, zorder=5)
    cbar = fig.colorbar(cl, ax=ax, orientation="horizontal", pad=0.01)
    cbar.set_ticks(np.arange(5/12, 5, 5/6))
    cbar.set_ticklabels(['TD', 'TS', 'STS', 'TY', 'VST', 'VT']) 
    cbar.ax.tick_params(direction='in', length=0, labelcolor='white', pad=-40)

    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_color('white')
        label.set_fontsize(30)

ax.scatter(lons[-1], lats[-1], s=200, color='black',marker = 'x' ,linewidths=5,transform = ccrs.PlateCarree(), zorder = 4)

ax.set_aspect(1, adjustable="datalim")

folder_path = rf"{base_dir}\bst_img\{storm['time'][0].strftime('%Y')}_{storm['name']}"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

plt.savefig(rf"{folder_path}\Whole_Track.png", bbox_inches = 'tight')
plt.close()