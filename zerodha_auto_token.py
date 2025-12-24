import time
import os
import pyotp
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
    """
    Cleans and pads the TOTP secret to ensure valid Base32 format.
    """
    if not key:
        return ""
    # Remove spaces/newlines and ensure uppercase
    key = key.strip().replace(" ", "").replace("\n", "").upper()
    
    # Check for invalid characters (Base32 uses A-Z and 2-7)
    # If it contains 0, 1, 8, 9 it might be the wrong key
    
    # Add padding if missing (Base32 requires length divisible by 8)
    missing_padding = len(key) % 8
    if missing_padding != 0:
        key += '=' * (8 - missing_padding)
    return key

def generate_token():
    print("🚀 Starting Auto-Login Script V2.0 (Debug Mode)...")
    
    # DEBUG: Check Secret Health (without revealing it)
    if not totp_secret:
        print("❌ CRITICAL: ZERODHA_TOTP_SECRET is missing or empty.")
        exit(1)
    else:
        print(f"ℹ️  TOTP Secret loaded. Length: {len(totp_secret)} chars.")

    # 1. Setup Headless Chrome
    print("📍 Step 0: Initializing Chrome...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        # 2. Get Login URL
        print("📍 Step 1: Navigating to Login Page...")
        kite = KiteConnect(api_key=api_key)
        login_url = kite.login_url()
        driver.get(login_url)
        
        # 3. Enter Credentials
        print("📍 Step 2: Entering User ID and Password...")
        wait = WebDriverWait(driver, 20)
        wait.until(EC.visibility_of_element_located((By.ID, "userid"))).send_keys(user_id)
        driver.find_element(By.ID, "password").send_keys(password)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        
        # 4. Handle TOTP
        print("📍 Step 3: Handling TOTP (2FA)...")
        
        # --- ROBUST TOTP GENERATION START ---
        clean_secret = clean_base32_key(totp_secret)
        print(f"ℹ️  Cleaned Secret Length: {len(clean_secret)}")
        
        try:
            totp = pyotp.TOTP(clean_secret)
            token_now = totp.now()
            print("✅ TOTP Token Generated Successfully.")
        except binascii.Error as e:
            print(f"❌ BASE32 DECODING ERROR: {e}")
            print("👉 Tip: Your TOTP Secret might contain invalid characters (0, 1, 8, 9) or is not a valid Base32 string.")
            print("👉 Action: Go to Zerodha -> Password & Security -> Enable TOTP -> 'Can't Scan? Copy Key' again.")
            raise e
        except Exception as e:
            print(f"❌ TOTP ERROR: {e}")
            raise e
        # --- ROBUST TOTP GENERATION END ---

        # Wait for the TOTP input field
        try:
            totp_field = wait.until(EC.visibility_of_element_located((By.XPATH, "//input[@type='text' and @minlength='6']")))
        except:
            totp_field = driver.find_element(By.ID, "userid")
        
        totp_field.send_keys(token_now)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        
        # 5. Wait for Redirect
        print("📍 Step 4: Waiting for Redirect...")
        try:
            wait.until(EC.url_contains("request_token"))
        except:
            print(f"❌ URL Timeout. Current URL: {driver.current_url}")
            try:
                err = driver.find_element(By.CLASS_NAME, "error").text
                print(f"❌ Zerodha Error Message: {err}")
            except: pass
            raise Exception("Login flow did not redirect to request_token.")

        current_url = driver.current_url
        
        if "request_token=" in current_url:
            request_token = current_url.split("request_token=")[1].split("&")[0]
            print(f"✅ Request Token Found: {request_token[:6]}...")
            
            # 6. Generate Access Token
            print("📍 Step 5: Exchanging for Access Token...")
            data = kite.generate_session(request_token, api_secret=api_secret)
            access_token = data["access_token"]
            
            # 7. Save to File
            with open("zerodha_token.txt", "w") as f:
                f.write(f"{api_key},{access_token}")
            
            print(f"🎉 SUCCESS! Token saved to zerodha_token.txt")
            
        else:
            raise Exception(f"URL did not contain request_token. Got: {current_url}")
            
    except Exception as e:
        print(f"❌ CRITICAL EXECUTION ERROR: {e}")
        exit(1)
    finally:
        driver.quit()

if __name__ == "__main__":
    generate_token()
