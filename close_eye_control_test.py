import subprocess
from monitorcontrol import get_monitors, PowerMode



def call():
    sentence = 'apbhcadnetfogmh_itjhkilemfn_okpiqdr_sAtpupvewasrysz'
    for i,e in enumerate(sentence):
        if i%2 !=0:
            print(e, end='')

monitors = get_monitors()

MAX_VOLUME = 65536

def setBrightSound(brightness, volume):
    # 모니터 밝기 조절
    if monitors:
        with monitors[0] as monitor:
            # 현재 밝기 확인
            current_brightness = monitor.get_luminance()
            print(f"현재 밝기: {current_brightness}%")

            # 밝기 설정 (0-100 사이의 값)
            new_brightness = brightness  # 원하는 밝기 값
            if brightness == 0:
                monitor.set_power_mode(PowerMode.off_soft)
            # if brightness == 100:
            #     monitor.set_luminance(new_brightness)
            #     monitor.set_power_mode(PowerMode.on)
            else:
                monitor.set_luminance(new_brightness)
            print(f"밝기를 {new_brightness}%로 설정했습니다.")
           
            
    else:
        print("모니터를 찾을 수 없습니다.")

    # 소리 조절
    # 0-100 볼륨 값에서 NirCmd의 범위로 변환
    volume_value = int(volume * 0.01 * MAX_VOLUME)  # 볼륨 값 계산 (정수 변환)
    
    # NirCmd 실행하여 볼륨 조절 (-volume_value로 소리 줄임)
    subprocess.run(["cmd", "/c", f"nircmd.exe setsysvolume {volume_value}"])
   

# 밝기 10%, 볼륨 40%줄임 설정
setBrightSound(100, 50)
call()





  