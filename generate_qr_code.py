# CassavaCare AI
# Author: Fusheini Abdul-Mumin <abdulmuminfusheini@gmail.com>
import qrcode
import socket
import os

def get_all_ip_addresses():
    """Get all non-loopback IPv4 addresses of this machine."""
    ips = []
    try:
        # Get machine name
        hostname = socket.gethostname()
        # Get list of all IPs linked to this machine
        addr_info = socket.getaddrinfo(hostname, None)
        for item in addr_info:
            ip = item[4][0]
            # Check for non-loopback IPv4
            if ":" not in ip and ip != "127.0.0.1":
                if ip not in ips:
                    ips.append(ip)
    except Exception as e:
        print(f"Error finding IPs: {e}")
    return ips

def generate_qr():
    ips = get_all_ip_addresses()
    
    if not ips:
        print("\n[ERROR] No local network connection found. Please connect to Wi-Fi.")
        return

    print("\n" + "="*40)
    print("  CassavaCare AI - QR Code Generator")
    print("="*40)
    print("Select your Wi-Fi interface IP address:")
    
    for i, ip in enumerate(ips):
        print(f"{i+1}. {ip}")
    
    try:
        user_input = input("\nEnter the number (e.g. 5) OR the IP address directly for your Wi-Fi: ").strip()
        
        selected_ip = None
        
        # Check if they entered one of the numbers from the list
        if user_input.isdigit():
            choice = int(user_input) - 1
            if 0 <= choice < len(ips):
                selected_ip = ips[choice]
        
        # Or if they entered the IP directly
        if not selected_ip and user_input in ips:
            selected_ip = user_input
            
        if not selected_ip:
            print("Invalid selection. Please choose a number from the list.")
            return
        
        url = f"http://{selected_ip}:5000"
        
        # Create QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        
        filename = "cassava_app_qr.png"
        img.save(filename)
        
        print("\n" + "="*40)
        print(f" SUCCESS!")
        print(f" Link: {url}")
        print(f" QR Code saved as: {filename}")
        print(" Show this image to users to scan with their phones.")
        print("="*40)
        
        # Try to open the file automatically (Windows)
        os.startfile(filename)
        
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_qr()
