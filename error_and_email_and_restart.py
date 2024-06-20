import os
import time
import smtplib
from email.mime.text import MIMEText
import subprocess
import signal
import datetime

# 10分钟内被修改的文件
def get_recently_modified_logs(folder_path, minutes=15):
    current_time = time.time()
    result = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            modified_time = os.path.getmtime(file_path)
            time_diff = current_time - modified_time
            if time_diff <= minutes * 60:
                result.append(file_path)

    return result


def kill_processes_related_to_log(file_path):
    # 使用lsof命令获取与日志文件相关的进程信息
    command = ['lsof', file_path]
    output = subprocess.check_output(command).decode('utf-8')

    # 解析lsof输出，提取PID
    pids = set()
    lines = output.split('\n')
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            pid = parts[1]
            pids.add(pid)

    # 使用os.kill()向每个PID发送SIGKILL信号
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed process with PID {pid}")
        except OSError as e:
            print(f"Failed to kill process with PID {pid}: {e}")

# 重启程序
def restart_program(root_folder_path, file_path):
    # 提取关键信息
    file_name = os.path.basename(file_path)
    info = file_name.split('_')
    court = info[0]
    has_record = 'record' in info

    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # 根据提取的信息确定要跳转的文件夹
    if has_record:
        folder_name = os.path.join(root_folder_path, f"{court}_record")
        new_log_file_name = f"{court}_{time_string}_record.log"
        command = f"cd {folder_name} && nohup python cap_record.py >~/LOG/{new_log_file_name}.log 2>&1 &"
        os.system(command)
    else:
        folder_name = os.path.join(root_folder_path, court)
        new_log_file_name = f"{court}_{time_string}.log"
        command = f"cd {folder_name} && nohup bash main.sh >~/LOG/{new_log_file_name}.log 2>&1 &"
        os.system(command)


# 设置文件路径和时间阈值（1分钟）
time_threshold = 120 # 单位为秒
root_folder_path = "/home/sportvision/"   # court1 court2 court1_record court2_record 所在目录
log_folder = os.path.join(root_folder_path, "LOG")

last_log_path = None
while True:
    latest_logs = get_recently_modified_logs(log_folder)
    for log_file in latest_logs:
        log_path = os.path.join(log_folder, log_file)
        # 获取文件的修改时间和当前时间，并计算它们之间的时间差
        file_mod_time = os.path.getmtime(log_path)
        current_time = time.time()
        time_diff = current_time - file_mod_time

        # 判断时间差是否小于时间阈值，如果是则打印信息，否则发送邮件
        if time_diff < time_threshold:
            print("File modified in the last 2 minutes.")
        elif last_log_path != log_path:
            # 设置邮件内容
            last_log_path = log_path
            mail_content = "The file {} has not been modified in the last 2 minutes.".format(log_path)
            mail_subject = "程序意外中止"

            # 发件人、收件人和邮件服务器配置
            from_address = "13066905418@163.com"
            to_address = "1257776077@qq.com"
            smtp_server = "smtp.163.com"
            # smtp_port = 587
            smtp_user = "13066905418@163.com"
            smtp_password = "OAYXRINPPKAIFSDI"

            # 创建邮件对象
            mail = MIMEText(mail_content)
            mail["Subject"] = mail_subject
            mail["From"] = from_address
            mail["To"] = to_address

            kill_processes_related_to_log(log_path)
            
            # 发送邮件
            try:
                smtp = smtplib.SMTP(smtp_server)
                smtp.ehlo()
                smtp.starttls()
                smtp.login(smtp_user, smtp_password)
                smtp.sendmail(from_address, to_address, mail.as_string())
                smtp.quit()
                print("Mail sent successfully.")
            except Exception as e:
                print("Error sending mail:", e)

            restart_program(root_folder_path, log_path)
