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
from selenium.webdriver.common.keys import Keys

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
    print("🚀 Starting Auto-Login Script V2.7 (TOTP Brute-Force)...")
    
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
        time.sleep(3) # Wait for TOTP page load
        
        # Check if we are on the TOTP page
        page_source = driver.page_source.lower()
        if "external totp" in page_source or "authenticator" in page_source:
            print("ℹ️  TOTP Page Detected.")
            
            # Strategy: Find ANY visible input field and type in it
            inputs = driver.find_elements(By.TAG_NAME, "input")
            typed = False
            for inp in inputs:
                try:
                    if inp.is_displayed() and inp.is_enabled():
                        inp_type = inp.get_attribute("type")
                        if inp_type in ["text", "number", "tel", "password"]:
                            inp.clear()
                            inp.send_keys(token_now)
                            print(f"✅ Typed TOTP into input (Type: {inp_type})")
                            typed = True
                            break # Assume only one main input on this page
                except: continue
            
            if not typed:
                print("❌ Could not find a visible input field for TOTP.")
                print(f"DEBUG: Page Inputs: {[i.get_attribute('outerHTML') for i in inputs]}")
                raise Exception("TOTP Input missing.")

            # Strategy: Find "Continue" button
            try:
                # Try finding button by text
                buttons = driver.find_elements(By.TAG_NAME, "button")
                clicked_btn = False
                for btn in buttons:
                    if "continue" in btn.text.lower() or "submit" in btn.type:
                        btn.click()
                        print(f"✅ Clicked button: '{btn.text}'")
                        clicked_btn = True
                        break
                
                if not clicked_btn:
                    # Fallback: Hit Enter in the input field
                    print("⚠️ Button not found. Hitting ENTER key...")
                    driver.switch_to.active_element.send_keys(Keys.RETURN)
            except Exception as e:
                print(f"⚠️ Button click failed: {e}")

        else:
            print("⚠️ Warning: Did not detect standard TOTP page text. Checking if already logged in...")

        
        # 4. Handle Post-Login Logic
        print("📍 Step 4: Waiting for Redirect/Authorize...")
        time.sleep(8) 
        
        # Check if still on Zerodha domain
        if "zerodha.com" in driver.current_url:
            print("ℹ️  Still on Zerodha domain. Investigating page content...")
            try:
                body_text = driver.find_element(By.TAG_NAME, "body").text.lower()
                
                if "invalid redirect" in body_text:
                    print("❌ ERROR: Zerodha says 'Invalid Redirect URL'.")
                    print("👉 FIX: Go to Kite Developer Console > My Apps > Edit App. Set Redirect URL to 'http://localhost'")
                    exit(1)
                
                if "authorize" in body_text or "allow" in body_text:
                    print("ℹ️  'Authorize' screen detected. Clicking submit...")
                    driver.find_element(By.XPATH, "//button[@type='submit']").click()
                    time.sleep(5)
            except: pass

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
                exit(1)
            except Exception as e:
                print(f"❌ API ERROR: {e}")
                exit(1)
            
        else:
            print(f"❌ Final URL: {current_url}")
            try:
                # Dump page text for debugging
                print("DEBUG PAGE DUMP:")
                print(driver.find_element(By.TAG_NAME, "body").text[:500])
            except: pass
            raise Exception("No request_token in URL.")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        exit(1)
    finally:
        driver.quit()

if __name__ == "__main__":
    generate_token()
