gtifrom selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

# Columns for the csv file
columns = ["Name", "Age", "Position", "Player Nationality", "Market Value", "Left", "LeagueLefted", "Joined", "LeagueJoined", "Fee"]

# Loading ChromeDriver with options
chromeOptions = Options()
chromeOptions.headless = False  # Set to True if you want to run it in headless mode
chromeOptions.add_argument('--disable-dev-shm-usage')  # Necessary for some Linux environments
chromeOptions.add_argument('--no-sandbox')  # Necessary for some Linux environments
chromeOptions.add_argument('--ignore-ssl-errors=yes')
chromeOptions.add_argument('--ignore-certificate-errors')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chromeOptions)
wait = WebDriverWait(driver, 10)

# Function to scrape the data
def main(): 
    while True:
        try:
            year = int(input(f"Enter Year of Transfers to Scrape (upto {time.localtime().tm_year}): "))
            if year > 1900 and year <= time.localtime().tm_year:
                break
            else:
                print("Please enter a valid year.")
        except ValueError:
            print("Invalid input. Please enter a numeric year.")
    
    accept_cookie = True
    player_data = []


    for page_id in range(1, 60):

        driver.implicitly_wait(10)
        
        url = f"https://www.transfermarkt.com/transfers/saisontransfers/statistik?land_id=0&ausrichtung=&spielerposition_id=&altersklasse=&leihe=&transferfenster=&saison-id={year}&plus=1&page={page_id}"
        driver.get(url)
        
        if accept_cookie:
            print("Waiting for cookie.....")
            try:
                # Dynamically locate iframe (if present)
                iframe = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "iframe[id^='sp_message_iframe']")))
                driver.switch_to.frame(iframe)
                cookie_popup = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.sp_choice_type_11')))
                cookie_popup.click()
                print("Cookie Accepted!")
                accept_cookie = False
                driver.switch_to.default_content()
            except Exception as e:
                print("No iframe found or cookie popup already accepted.")
        
        try:
            # Scrape the data
            rankings = driver.find_element(By.ID, 'yw0')
            rows = rankings.find_elements(By.CSS_SELECTOR, '.odd, .even')
            print(f"Page: {page_id} Done.")

            for index, row in enumerate(rows):
                player_data.append(get_player_details(row))
        
        except Exception as e:
            print(f"Error on page {page_id}: {str(e)}")
        
    # Write the data to a CSV file
    df = pd.DataFrame(data=player_data, columns=columns)
    df.to_csv(f"transfermarkt_{year}.csv", index=False)
    print(f"Data saved to transfermarkt_{year}.csv")
    
    driver.close()

# Function to get the player details
def get_player_details(row):
    try:
        details = row.text.split('\n')
        contents = {}
        contents["Name"] = details[1].encode('utf-8').decode('utf-8')
        contents["Position"] = details[2]  
        contents["Age"] = int(details[3].split(' ')[0])
        market_value = details[3].split(' ')[1]
        contents["Market Value"] = extract_numerical_value(market_value)
        contents["Left"] = details[4].encode('utf-8').decode('utf-8')
        contents["LeagueLefted"] = details[5].encode('utf-8').decode('utf-8')
        contents["Joined"] = details[6]

        fee = details[8]
        contents["Fee"] = numerical_value_and_transfer_category(fee, contents)
        country_flags = row.find_elements(By.CSS_SELECTOR, 'img.flaggenrahmen')
        nationality = country_flags[0].get_attribute('title')
        contents["Player Nationality"] = nationality
        contents["LeagueJoined"] = details[7].encode('utf-8').decode('utf-8')
        return contents
    except Exception as e:
        print(f"Error processing row: {str(e)}")
        return {}

# Function to extract the numerical value
def extract_numerical_value(value):
    numerical_part = ''.join(filter(str.isdigit, value))
    if numerical_part:
        return float(numerical_part)/100
    else:
        return None  # or assign a default value, e.g., 0

# Function to extract the numerical value and transfer category
def numerical_value_and_transfer_category(value, contents):
    numerical_part = ''.join(filter(str.isdigit, value))
    value = value.lower()
    if numerical_part:
        contents["Transfer Type"] = "Permanent"
        return float(numerical_part)/100
    else:
        if 'loan' in value:
            contents["Transfer Type"] = "Loan"
            return None
        elif value == 'free transfer':
            contents["Transfer Type"] = "Free"
            return None
        else:
            contents["Transfer Type"] = "Unknown"
            return None

if __name__ == "__main__":
    main()

