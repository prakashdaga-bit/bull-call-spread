import time
import os
import pyotp
from kiteconnect import KiteConnect
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ==========================================
# CONFIGURATION (FROM GITHUB SECRETS)
# ==========================================
api_key = os.environ.get("ZERODHA_API_KEY")
api_secret = os.environ.get("ZERODHA_API_SECRET")
user_id = os.environ.get("ZERODHA_USER_ID")
password = os.environ.get("ZERODHA_PASSWORD")
totp_secret = os.environ.get("ZERODHA_TOTP_SECRET")

def generate_token():
    print("🚀 Starting Auto-Login via GitHub Actions...")
    
    # 1. Setup Headless Chrome (Optimized for CI/CD)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        # 2. Get Login URL
        kite = KiteConnect(api_key=api_key)
        login_url = kite.login_url()
        driver.get(login_url)
        
        # 3. Enter Credentials
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.ID, "userid"))).send_keys(user_id)
        driver.find_element(By.ID, "password").send_keys(password)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        
        # 4. Handle TOTP
        wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='text']"))) # Wait for TOTP field
        totp = pyotp.TOTP(totp_secret)
        token_now = totp.now()
        
        driver.find_element(By.XPATH, "//input[@type='text']").send_keys(token_now)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        
        # 5. Wait for Redirect and Grab Request Token
        time.sleep(5) 
        current_url = driver.current_url
        
        if "request_token=" in current_url:
            request_token = current_url.split("request_token=")[1].split("&")[0]
            print(f"✅ Request Token Retrieved")
            
            # 6. Generate Access Token
            data = kite.generate_session(request_token, api_secret=api_secret)
            access_token = data["access_token"]
            
            # 7. Save to File
            with open("zerodha_token.txt", "w") as f:
                f.write(f"{api_key},{access_token}")
            
            print(f"🎉 Token Generated Successfully")
            
        else:
            raise Exception("URL did not contain request_token. Login failed.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1) # Fail the workflow if this happens
    finally:
        driver.quit()

if __name__ == "__main__":
    generate_token()
