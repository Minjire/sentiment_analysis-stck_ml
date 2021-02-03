# get activated on some specific time
# get the news headlines
# send the to sentimental analysis model
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

chrome_options = webdriver.ChromeOptions()
chrome_options.binary_location = os.environ.get("GOOGLE_CHROME_BIN")
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--no-sandbox")

# initializing driver
ChromeDriver = webdriver.Chrome(executable_path=os.environ.get("CHROMEDRIVER_PATH"), chrome_options=chrome_options)
ChromeDriver.get("https://kenyanwallstreet.com/#")
ChromeDriver.maximize_window()

wait = WebDriverWait(ChromeDriver, 5)
presence = EC.presence_of_element_located
visible = EC.visibility_of_element_located
wait.until(visible((By.TAG_NAME, "iframe")))
ChromeDriver.switch_to.frame(ChromeDriver.find_element_by_tag_name("iframe"))
ChromeDriver.refresh()
companies = ['safaricom', 'equity', 'eabl', 'kcb']

for company in companies:
    # searching new article for each company
    wait.until(visible((By.NAME, "s")))
    ChromeDriver.find_element_by_name('s').send_keys(company)
    wait.until(visible((By.CLASS_NAME, "jeg_search_button")))
    ChromeDriver.find_element_by_class_name("jeg_search_button").click()

    # get today's article
    wait.until(visible((By.CLASS_NAME, "jnews_search_content_wrapper")))
    container = ChromeDriver.find_element_by_class_name('jeg_posts')
    articles = container.find_elements_by_tag_name('article')
    date = articles[0].find_element_by_class_name('jeg_meta_date').text
    print(date)
