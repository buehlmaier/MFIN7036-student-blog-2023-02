---
Title: How to Install Chromedriver in macOS (by Group "NLP Intelligence")
Date: 2023-03-08 12:00
Category: Progress Report
---

By Group NLP Intelligence

When our group was doing web scraping, we downloaded Chromedriver and tried to use it to control Chrome website. However, the code of running the process we knew fitted for Windows only (as following).

```python
import selenium
from selenium import webdriver

browser = webdriver.Chrome('path/chromedriver.exc')

```

For two members who use OS system, they had to fixed several errors such as the following:
![Picture showing P1]({static}/images/NLP-Intelligence-Post01_P1.jpeg)

Or something warns that the file does not have permission. 

Although the solution the issue is not difficult, it took time to figure it out. We hope this blog can help others who meet the similar problem and save their time.

## Step 1: Drag Chromedriver File to /usr/local/bin

If your `usr/local/bin` file is visible in Finder, then congratulations because you just need to do one easy thing: drag Chromedriver file you downloaded into your `usr/local/bin` file. If it is hidden in your computer, then you will have to take following steps. 

## Step 2: Find /usr/local/bin in Terminal

Open your Terminal first, and type `cd /usr/local/bin/` to enter it. 

## Step 3: Move Chromedriver to /usr/local/bin

Type `mv ./chromedriver_mac64/chromedriver /usr/local/bin/` to move Chromedrive from original location to `/usr/local/bin`.

## Step 4: Give Your Permission

If there is an error in your code showing the file does not have permission to execute, you could go to System Preferences -> Security&Privacy -> General. Then, click the lock to make changes. Under the section of Allow apps downloaded from, you could choose "App Store and identified developers." Sometimes, there is another request there asking you whether to allow the access from chromedriver. You could just tap "Allow" there.

![Picture showing P2]({static}/images/NLP-Intelligence-Post01_P2.jpeg)

Finally, the code of running Chromedriver for OS system is 

```python
import selenium
from selenium import webdriver

browser = webdriver.Chrome('/usr/local/bin/chromedriver')

```