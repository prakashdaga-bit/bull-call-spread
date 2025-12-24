import time
import os
import pyotp
import base64
import binascii
from kiteconnect import KiteConnect
import kiteconnect.exceptions
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ==========================================
# CONFIGURATION
# ==========================================
api_key = os.environ.get("ZERODHA_API_KEY")
api_secret = os.environ.get("ZERODHA_API_SECRET")
user_id = os.environ.get("ZERODHA_USER_ID")
password = os.environ.get("ZERODHA_PASSWORD")
totp_secret = os.environ.get("ZERODHA_TOTP_SECRET")

def clean_base32_key(key):
    if not key: return ""
    key = key.strip().replace(" ", "").replace("\n", "").upper()
    missing_padding = len(key) % 8
    if missing_padding != 0:
        key += '=' * (8 - missing_padding)
    return key

def generate_token():
    print("🚀 Starting Auto-Login Script V2.5 (Secret Debugger)...")
    
    # 1. Setup Data
    if not totp_secret:
        print("❌ CRITICAL: ZERODHA_TOTP_SECRET is missing.")
        exit(1)
    
    raw_secret = clean_base32_key(totp_secret)
    token_now = None
    try:
        totp = pyotp.TOTP(raw_secret)
        token_now = totp.now()
        print(f"✅ TOTP Generated. Secret length: {len(raw_secret)}")
    except Exception as e:
        print(f"❌ TOTP Error: {e}")
        exit(1)

    # 2. Setup Chrome
    print("📍 Step 0: Initializing Chrome...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        # 3. Login
        print("📍 Step 1: Navigating to Login Page...")
        kite = KiteConnect(api_key=api_key)
        login_url = kite.login_url()
        driver.get(login_url)
        
        print("📍 Step 2: Entering User ID and Password...")
        wait = WebDriverWait(driver, 20)
        wait.until(EC.visibility_of_element_located((By.ID, "userid"))).send_keys(user_id)
        driver.find_element(By.ID, "password").send_keys(password)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        
        print("📍 Step 3: Entering TOTP...")
        try:
            # Flexible wait for any input field
            totp_field = wait.until(EC.visibility_of_element_located((By.XPATH, "//input[@type='text' or @type='number' or @type='tel']")))
        except:
            # Fallback for weird ID reuse
            totp_field = driver.find_element(By.ID, "userid")
        
        totp_field.send_keys(token_now)
        
        # Robust button click
        try:
            driver.find_element(By.XPATH, "//button[@type='submit']").click()
        except:
            driver.execute_script("document.querySelector('button[type=submit]').click();")
        
        # 4. Handle Post-Login Logic
        print("📍 Step 4: Waiting for Redirect/Authorize...")
        time.sleep(5) 
        
        # Check if stuck on Authorize screen
        if "zerodha.com" in driver.current_url:
            if "Authorize" in driver.page_source:
                print("ℹ️  'Authorize' screen detected. Clicking...")
                try:
                    driver.find_element(By.XPATH, "//button[@type='submit']").click()
                    print("✅ Clicked Authorize.")
                    time.sleep(3)
                except:
                    print("⚠️ Could not click Authorize automatically.")

        # 5. Extract Token
        current_url = driver.current_url
        
        if "request_token=" in current_url:
            request_token = current_url.split("request_token=")[1].split("&")[0]
            print(f"✅ Request Token Found: {request_token[:6]}...")
            
            print("📍 Step 5: Generating Access Token...")
            try:
                data = kite.generate_session(request_token, api_secret=api_secret)
                access_token = data["access_token"]
                
                with open("zerodha_token.txt", "w") as f:
                    f.write(f"{api_key},{access_token}")
                
                print(f"🎉 SUCCESS! Token saved.")
                
            except kiteconnect.exceptions.TokenException as e:
                print(f"❌ AUTH ERROR: {e}")
                print("👉 MOST LIKELY CAUSE: Your 'ZERODHA_API_SECRET' in GitHub Secrets is incorrect.")
                print("👉 Action: Check Developer Portal -> My Apps -> API Secret. It is NOT the same as API Key.")
                exit(1)
            except Exception as e:
                print(f"❌ API ERROR: {e}")
                exit(1)
            
        else:
            print(f"❌ Final URL: {current_url}")
            try:
                # Debug print only if failed
                print(f"Debug Title: {driver.title}")
            except: pass
            raise Exception("No request_token in URL.")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        exit(1)
    finally:
        driver.quit()

if __name__ == "__main__":
    generate_token()
