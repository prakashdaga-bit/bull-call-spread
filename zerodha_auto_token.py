import time
import os
import pyotp
import base64
import binascii
from kiteconnect import KiteConnect
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
    print("🚀 Starting Auto-Login Script V2.3 (Deep Debug)...")
    
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
            totp_field = wait.until(EC.visibility_of_element_located((By.XPATH, "//input[@type='text' and @minlength='6']")))
        except:
            totp_field = driver.find_element(By.ID, "userid") # Fallback
        
        totp_field.send_keys(token_now)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        
        # 4. Handle Post-Login Logic
        print("📍 Step 4: Waiting for Redirect/Authorize...")
        time.sleep(5) # Allow page to load/redirect
        
        # Check if we are still on the Zerodha domain (means redirect didn't happen yet)
        if "zerodha.com" in driver.current_url:
            print("ℹ️  Still on Zerodha domain. Checking page content...")
            print(f"📄 Page Title: {driver.title}")
            
            # Look for Authorize Button explicitly
            buttons = driver.find_elements(By.TAG_NAME, "button")
            clicked = False
            for btn in buttons:
                if "authorize" in btn.text.lower():
                    print(f"✅ Found Authorize Button: '{btn.text}'. Clicking...")
                    btn.click()
                    clicked = True
                    time.sleep(3) # Wait for redirect after click
                    break
            
            if not clicked:
                # If no button found, print visible text to debug errors (like "Invalid Redirect URL")
                body_text = driver.find_element(By.TAG_NAME, "body").text
                print(f"⚠️  STUCK. Visible Page Text Snippet:\n{body_text[:300]}...")

        # 5. Extract Token
        current_url = driver.current_url
        print(f"🔎 Checking URL: {current_url}")
        
        if "request_token=" in current_url:
            request_token = current_url.split("request_token=")[1].split("&")[0]
            print(f"✅ Request Token Found: {request_token[:6]}...")
            
            print("📍 Step 5: Generating Access Token...")
            data = kite.generate_session(request_token, api_secret=api_secret)
            access_token = data["access_token"]
            
            with open("zerodha_token.txt", "w") as f:
                f.write(f"{api_key},{access_token}")
            
            print(f"🎉 SUCCESS! Token saved.")
        else:
            raise Exception("Final URL does not contain request_token. Check Redirect URL setting in Kite Console.")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        exit(1)
    finally:
        driver.quit()

if __name__ == "__main__":
    generate_token()
