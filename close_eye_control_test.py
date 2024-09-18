import subprocess

# 눈이 감겼을 때 밝기랑 조절하는 함수
def close_eye(level, volume):
    # 밝기 조절하는 powershell 명령어 -> 그냥 실행 가능
    setMonitor = f"$monitor = Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods\n"
    setBrightness = f"$monitor.WmiSetBrightness(1, %d)\n" % level

    # 소리 조절하는 powershell 명령어 -> nircmd 파일 필요
    volume_value = int(65535 * (volume / 100))
    # nircmd가 있는 경로 입력 => 체험존(nircmd.exe파일 옮긴 다음에 그 경로 여기에 입력)
    nircmd_path = r"C:\Users\luna2\Downloads\nircmd-x64\nircmd.exe"
    setSound = nircmd_path + " setsysvolume " + str(volume_value)

    # powershell 명령어 실행
    p = subprocess.run(["powershell.exe", "-Command", setMonitor + setBrightness + setSound], capture_output=True, text=True)

# 인자 순서는 (밝기, 소리)
close_eye(50, 40)
