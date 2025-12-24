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
    """
    Cleans and pads the TOTP secret to ensure valid Base32 format.
    """
    if not key:
        return ""
    key = key.strip().replace(" ", "").replace("\n", "").upper()
    missing_padding = len(key) % 8
    if missing_padding != 0:
        key += '=' * (8 - missing_padding)
    return key

def generate_token():
    print("🚀 Starting Auto-Login Script V2.2 (Authorize Handler)...")
    
    if not totp_secret:
        print("❌ CRITICAL: ZERODHA_TOTP_SECRET is missing.")
        exit(1)
    
    raw_secret = totp_secret.strip().replace(" ", "").replace("\n", "").upper()
    print(f"ℹ️  Loaded Secret Length: {len(raw_secret)} chars")

    # TOTP Generation
    token_now = None
    for i in range(7):
        try:
            candidate_secret = raw_secret + ('=' * i)
            base64.b32decode(candidate_secret, casefold=True)
            totp = pyotp.TOTP(candidate_secret)
            token_now = totp.now()
            print(f"✅ TOTP Generated Successfully (Padding: {i} char(s)).")
            break
        except: continue

    if not token_now:
        print("❌ CRITICAL: Could not decode TOTP Secret.")
        exit(1)

    # Setup Chrome
    print("📍 Step 0: Initializing Chrome...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    # Add user-agent to avoid some bot detection
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        # Login
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
            # Check for error immediately
            try:
                err = driver.find_element(By.CLASS_NAME, "error").text
                print(f"❌ Login Error (Pre-TOTP): {err}")
                raise Exception("Login failed before TOTP.")
            except:
                totp_field = driver.find_element(By.ID, "userid")
        
        totp_field.send_keys(token_now)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        
        print("📍 Step 4: Waiting for Redirect/Authorize...")
        time.sleep(3) # Wait for page transition
        
        # Check for error on TOTP page (e.g. Invalid TOTP)
        try:
            err_elem = driver.find_element(By.CLASS_NAME, "error")
            if err_elem.is_displayed():
                print(f"❌ Zerodha Error after TOTP: {err_elem.text}")
                print("👉 Possible Causes: Incorrect TOTP Secret, Time Sync Issue, or Account Locked.")
                raise Exception(f"Zerodha UI Error: {err_elem.text}")
        except: pass

        # Check for "Authorize" screen
        if "Authorize" in driver.title or "Authorize" in driver.page_source:
            print("ℹ️  'Authorize' screen detected. Attempting to click...")
            try:
                # Try finding a button with type submit inside a form-footer or just the main button
                auth_btn = driver.find_element(By.XPATH, "//button[@type='submit']")
                auth_btn.click()
                print("✅ Clicked Authorize.")
            except Exception as e:
                print(f"⚠️  Could not click Authorize: {e}")

        # Wait for request_token in URL
        try:
            wait.until(EC.url_contains("request_token"))
        except:
            print(f"❌ URL Timeout. Current URL: {driver.current_url}")
            # print(f"DEBUG: Page Source:\n{driver.page_source[:500]}...") # Verify what's on page
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
