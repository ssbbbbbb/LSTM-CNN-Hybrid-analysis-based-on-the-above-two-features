import argparse
import os
from scapy.all import rdpcap, hexdump, IP, TCP, UDP


def process_pcap_file(input_pcap, output_txt):
    """
    讀取一個 .pcap 文件，提取封包的五元組並將十六進制表示寫入輸出文件。

    :param input_pcap: 要處理的 .pcap 文件路徑
    :param output_txt: 十六進制輸出文件的儲存路徑
    """
    try:
        # 讀取封包文件
        packets = rdpcap(input_pcap)
        print(f"成功讀取 {len(packets)} 個封包 from {input_pcap}")
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {input_pcap}")
        return
    except Exception as e:
        print(f"讀取封包時發生錯誤 ({input_pcap}): {e}")
        return

    try:
        # 開啟輸出文件
        with open(output_txt, "w") as f:
            # 遍歷封包並處理
            for idx, pkt in enumerate(packets, start=1):
                if pkt.haslayer(IP):
                    src_ip = pkt[IP].src
                    dst_ip = pkt[IP].dst
                    protocol_num = pkt[IP].proto  # 協議號 (6 = TCP, 17 = UDP)

                    # 提取 TCP 或 UDP 端口
                    if pkt.haslayer(TCP):
                        src_port = pkt[TCP].sport
                        dst_port = pkt[TCP].dport
                        protocol_name = "TCP"
                    elif pkt.haslayer(UDP):
                        src_port = pkt[UDP].sport
                        dst_port = pkt[UDP].dport
                        protocol_name = "UDP"
                    else:
                        src_port = "-"
                        dst_port = "-"
                        protocol_name = "OTHER"

                    # 寫入五元組資訊
                    f.write(f"封包 {idx}: {src_ip}:{src_port} -> {dst_ip}:{dst_port} ({protocol_name})\n")

                    # 將封包以十六進制格式寫入輸出文件
                    hexdump_output = hexdump(pkt, dump=True)
                    f.write(hexdump_output + "\n")
                    f.write("-" * 80 + "\n")  # 分隔符
        print(f"十六進制輸出已儲存至 {output_txt}")
    except Exception as e:
        print(f"寫入十六進制文件時發生錯誤 ({output_txt}): {e}")


def batch_process(input_dir, output_dir):
    """
    批量處理指定目錄中的所有 .pcap 文件。

    :param input_dir: 輸入目錄，包含要處理的 .pcap 文件
    :param output_dir: 輸出目錄，用於儲存十六進制輸出文件
    """
    # 檢查輸入目錄是否存在
    if not os.path.isdir(input_dir):
        print(f"錯誤：輸入目錄 {input_dir} 不存在或不是目錄。")
        return

    # 創建輸出目錄（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 遍歷輸入目錄中的所有 .pcap 文件
    pcap_files = [f for f in os.listdir(input_dir) if f.endswith('.pcap')]

    if not pcap_files:
        print(f"在目錄 {input_dir} 中找不到任何 .pcap 文件。")
        return

    print(f"在目錄 {input_dir} 中找到 {len(pcap_files)} 個 .pcap 文件。開始處理...")

    for pcap_file in pcap_files:
        input_pcap = os.path.join(input_dir, pcap_file)
        base_name = os.path.splitext(pcap_file)[0]
        output_txt = os.path.join(output_dir, f"{base_name}_hex.txt")

        print(f"\n處理文件: {input_pcap}")
        process_pcap_file(input_pcap, output_txt)


def main():
    # 設置命令行參數解析
    parser = argparse.ArgumentParser(description="批量讀取 .pcap 檔案並將封包以十六進制格式輸出")
    parser.add_argument(
        "-i", "--input_dir",
        required=True,
        help="/home/username/pcap/bettersurf"
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="/home/username/pcap/16進/bettersurf"
    )

    args = parser.parse_args()

    # 呼叫批量處理函數
    batch_process(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()






