from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import date
from csv import writer

import os

# chrome_options = webdriver.ChromeOptions()
# chrome_options.binary_location = os.environ.get("GOOGLE_CHROME_BIN")
# chrome_options.add_argument("--headless")
# chrome_options.add_argument("--disable-dev-shm-usage")
# chrome_options.add_argument("--no-sandbox")

# initializing driver
ChromeDriver = webdriver.Chrome()
ChromeDriver.get("https://kenyanwallstreet.com/#")
ChromeDriver.maximize_window()

wait = WebDriverWait(ChromeDriver, 5)
presence = EC.presence_of_element_located
visible = EC.visibility_of_element_located
wait.until(visible((By.TAG_NAME, "iframe")))
ChromeDriver.switch_to.frame(ChromeDriver.find_element_by_tag_name("iframe"))
ChromeDriver.refresh()
today = date.today()
# Textual month, day and year
today = today.strftime("%B %d, %Y")
today.upper()
companies = ['safaricom', 'equity', 'eabl', 'kcb']

for company in companies:
    # searching new article for each company
    wait.until(visible((By.NAME, "s")))
    ChromeDriver.find_element_by_name('s').send_keys(company)
    wait.until(visible((By.CLASS_NAME, "jeg_search_button")))
    ChromeDriver.find_element_by_class_name("jeg_search_button").click()
    updated_data = []
    # get today's article
    wait.until(visible((By.CLASS_NAME, "jnews_search_content_wrapper")))
    container = ChromeDriver.find_element_by_class_name('jeg_posts')
    articles = container.find_elements_by_tag_name('article')
    date = articles[0].find_element_by_class_name('jeg_meta_date').text

    if 'JANUARY 30, 2021' == date:
        header = articles[0].find_element_by_class_name('jeg_post_title').text
        updated_data.append([header, date])
        path = 'data/raw/test.csv'
        with open(path, 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(updated_data)
            f_object.close()
    else:
        print('no update')
    ChromeDriver.find_element_by_name('s').clear()



