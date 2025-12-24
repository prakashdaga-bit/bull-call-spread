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

def generate_token():
    print("🚀 Starting Auto-Login Script V2.1 (Padding Fixer)...")
    
    # 1. Validate Secret Health
    if not totp_secret:
        print("❌ CRITICAL: ZERODHA_TOTP_SECRET is missing.")
        exit(1)
    
    # Remove spaces/formatting
    raw_secret = totp_secret.strip().replace(" ", "").replace("\n", "").upper()
    print(f"ℹ️  Loaded Secret Length: {len(raw_secret)} chars")
    
    if len(raw_secret) not in [16, 32]:
        print(f"⚠️  WARNING: Standard Base32 secrets are usually 16 or 32 chars. Yours is {len(raw_secret)}.")
        print("👉  If login fails, please re-copy the secret from Zerodha carefully.")

    # 2. TOTP Generation with Brute-Force Padding
    token_now = None
    
    # Try adding padding from 0 to 6 equals signs
    # Base32 requires length divisible by 8.
    for i in range(7):
        try:
            candidate_secret = raw_secret + ('=' * i)
            # Validate format by attempting decode
            base64.b32decode(candidate_secret, casefold=True)
            
            # If decode works, generate token
            totp = pyotp.TOTP(candidate_secret)
            token_now = totp.now()
            print(f"✅ TOTP Generated Successfully (Padding: {i} char(s)).")
            break
        except binascii.Error:
            continue # Try next padding length
        except Exception:
            continue

    if not token_now:
        print("❌ CRITICAL: Could not decode TOTP Secret with any padding.")
        print("👉 The secret key likely contains invalid characters or is truncated.")
        print("👉 Action: Go to Zerodha -> Password & Security -> Disable/Enable TOTP -> Copy new key.")
        exit(1)

    # 3. Setup Headless Chrome
    print("📍 Step 0: Initializing Chrome...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        # 4. Login Flow
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
            totp_field = driver.find_element(By.ID, "userid")
        
        totp_field.send_keys(token_now)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        
        print("📍 Step 4: Waiting for Redirect...")
        try:
            wait.until(EC.url_contains("request_token"))
        except:
            print(f"❌ Timeout. Current URL: {driver.current_url}")
            try:
                err = driver.find_element(By.CLASS_NAME, "error").text
                print(f"❌ Zerodha UI Error: {err}")
            except: pass
            raise Exception("No redirect to request_token.")

        current_url = driver.current_url
        
        if "request_token=" in current_url:
            request_token = current_url.split("request_token=")[1].split("&")[0]
            print(f"✅ Request Token: {request_token[:6]}...")
            
            print("📍 Step 5: Generating Access Token...")
            data = kite.generate_session(request_token, api_secret=api_secret)
            access_token = data["access_token"]
            
            with open("zerodha_token.txt", "w") as f:
                f.write(f"{api_key},{access_token}")
            
            print(f"🎉 SUCCESS! Token saved.")
            
        else:
            raise Exception("URL missing request_token.")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        exit(1)
    finally:
        driver.quit()

if __name__ == "__main__":
    generate_token()
