import pyttsx3 as p
import speech_recognition as sr
from googletrans import Translator
import pyjokes
import requests
import random
import webbrowser
import datetime
import csv
import re
from datetime import datetime
import pywhatkit
import wikipedia
import webbrowser
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
import os
# Load the model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Chatbot conversation
chat_history_ids = None

warnings.filterwarnings("ignore", category=UserWarning)

CSV_FILE = 'expenses.csv'

flashcards = []
correct_answers = 0  
total_questions = 0 
engine = p.init()
engine.setProperty('rate', 180)

currency_mappings = {
    "rupees": "rupees",
    "rs": "rupees",
    "dollars": "dollars",
    "$": "dollars"
}

def speak(text):
    engine.say(text)
    engine.runAndWait()

def calibrate_microphone(device_index=None):
    try:
        # Re-initialize the recognizer and microphone to ensure it's fresh
        r = sr.Recognizer()
        with sr.Microphone(device_index=device_index) as source:
            engine.say("Calibrating microphone. Please be quiet.")
            engine.runAndWait()
            r.adjust_for_ambient_noise(source, duration=5)
            print("Microphone calibrated.")
    except OSError as e:
        print(f"An error occurred during microphone calibration: {e}")
        engine.say("An error occurred while accessing the microphone. Please check your microphone settings.")
        engine.runAndWait()
        return False
    return True

def process_voice_input_calculator():
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio)
        print(f"User said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Speech recognition service is down"

def listen_for_number(prompt):
    """Listen for a number from the user and return it."""
    engine.say(prompt)
    engine.runAndWait()
    
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        print("Listening for a number...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            print("Recognizing...")
            number = r.recognize_google(audio)
            # Debugging: print recognized number
            print(f"Recognized number: {number}")
            return number.lower()  # Return as lowercase for uniformity
        except sr.WaitTimeoutError:
            speak("Listening timed out.")
        except sr.UnknownValueError:
            speak("Could not understand the audio.")
        except sr.RequestError:
            speak("Speech recognition service is unavailable.")
        return None
    
def process_expense_input():
    with sr.Microphone() as source:
        print("Listening for your expense...")
        # Adjust for ambient noise before listening
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, timeout=5)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio)
        print(f"User said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return "Could not understand audio"
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}")
        return "Service is down"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "Error"
def recognize_audio():
    """Recognize speech from the microphone and return it as text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError:
        print("Could not request results from the speech recognition service.")
        return None
    
def process_voice_input(prompt):
    engine.say(prompt)
    engine.runAndWait()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            print("Recognizing...")
            query = r.recognize_google(audio)
            print(f"User said: {query}")
            return query.lower()
        except sr.WaitTimeoutError:
            print("Listening timed out. Please try again.")
            return "timeout"
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return "unknown"
        except sr.RequestError:
            print("Speech recognition service is down.")
            return "service error"
        except Exception as e:
            print(f"An error occurred: {e}")
            return "error"
        
def takeCommand():
    """Function to take voice input from the user."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening.....")
        r.pause_threshold = 1
        r.energy_threshold = 300
        audio = r.listen(source, 0, 4)
    try:
        print("Understanding...")
        query = r.recognize_google(audio, language='en-in')
        print(f"You Said: {query}\n")
    except Exception as e:
        print("Say that again, please.")
        return "None"
    return query


def reply(user_input, chat_history_ids):
    """Generate a response based on user input and return updated chat history."""
    # Encode the new user input, add the EOS token, and return a tensor in PyTorch
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input to the chat history
    if chat_history_ids is None:
        chat_history_ids = new_input_ids
    else:
        chat_history_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)

    # Generate a response from the model
    response_ids = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, 
                                   no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Get the chatbot's response and decode it
    chatbot_response = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Chatbot:", chatbot_response)
    return chatbot_response, chat_history_ids 

def chat_with_bot(initial_message):
    """Chat with the bot in a loop, starting with an initial message provided by the user."""
    chat_history_ids = None

    # Print the initial message
    print(f"You: {initial_message}")
    chatbot_response, chat_history_ids = reply(initial_message, chat_history_ids)  # Call reply for initial message
    speak(chatbot_response)  # Speak the chatbot's initial response
    return chatbot_response

def searchGoogle(query):
    """Function to perform a Google search."""
    query = query.replace("google", "").strip()
    speak("This is what I found on Google.")
    try:
        pywhatkit.search(query)
        result = wikipedia.summary(query, sentences=1)
        speak(result)
    except:
        speak("No speakable output is available.")

def searchYoutube(query):
    """Function to perform a YouTube search."""
    query = query.replace("youtube", "").strip()
    speak("This is what I found for your search on YouTube.")
    web = f"https://www.youtube.com/results?search_query={query}"
    webbrowser.open(web)
    pywhatkit.playonyt(query)
    speak("Done.")

def searchWikipedia(query):
    """Function to search Wikipedia for a summary."""
    query = query.replace("wikipedia", "").strip()
    speak("Searching on Wikipedia...")
    try:
        results = wikipedia.summary(query, sentences=2)
        speak("According to Wikipedia,")
        print(results)
        speak(results)
    except wikipedia.exceptions.DisambiguationError as e:
        speak("There are multiple results for this query. Please specify more details.")
    except Exception as e:
        speak("No results were found on Wikipedia.")

    
def quiz_mode():
    global correct_answers, total_questions

    if not flashcards:
        engine.say("No flashcards available. Please add some first.")
        engine.runAndWait()
        return

    engine.say("Starting the quiz. Answer the questions with your voice. Say 'exit' to quit, or say 'score' to view your score.")
    engine.runAndWait()

    random.shuffle(flashcards)

    for question, correct_answer in flashcards:
        engine.say(question)
        engine.runAndWait()

        user_answer = process_voice_input("Your answer.")
        
        if user_answer in ["exit", "timeout", "unknown", "service error", "error"]:
            engine.say("Exiting quiz mode.")
            engine.runAndWait()
            break
        
        total_questions += 1  # Increment total questions count
        if user_answer == correct_answer:
            correct_answers += 1  # Increment correct answers count
            engine.say("Correct!")
        else:
            engine.say(f"Incorrect. The correct answer is {correct_answer}.")
        
        engine.runAndWait()

    # Quiz over message
    engine.say("Quiz is over. Thank you for participating!")
    
    # Provide score feedback
    view_score()  # Call view_score function to announce the score
    engine.runAndWait()

ops = {
    'plus': '+',
    '+': '+',
    'minus': '-',
    '-': '-',
    'times': '*',
    '*': '*',
    'multiply': '*',
    'divided by': '/',
    '/': '/',
    'divide': '/',
    'modulus': '%',
    '%': '%',
}

def evaluate_expression(expression):
    try:
        # Replace words with their corresponding symbols
        expression = expression.replace("plus", "+").replace("minus", "-")\
                               .replace("times", "*").replace("multiply", "*")\
                               .replace("divided by", "/").replace("divide", "/")\
                               .replace("modulus", "%")

        # Evaluate the mathematical expression
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error: {str(e)}"
    
def add_expense(amount, currency, category, description):
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), amount, currency, category, description])

def parse_expense_details(input_text):
    try:
        # Regular expression to extract amount, currency, category, and description
        match = re.search(r'(\$|rs|rupees|dollars)\s*(\d+\.?\d*)\s*for\s*([a-zA-Z\s]+)(?:\s*-\s*(.*))?', input_text)
        if match:
            currency = match.group(1).lower()
            amount = float(match.group(2))
            currency = currency_mappings.get(currency, currency)  # Convert abbreviation to full currency name
            category = match.group(3).strip().lower()  # Allow spaces in category names
            description = match.group(4).strip() if match.group(4) else "No description"

            # Log parsed details for debugging
            print(f"Parsed details: Amount: {amount}, Currency: {currency}, Category: {category}, Description: {description}")

            return amount, currency, category, description
        else:
            print("No match found in input.")
            return None
    except Exception as e:
        print(f"Error parsing input: {e}")
        return None

def show_expenses():
    try:
        with open(CSV_FILE, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            expenses = list(reader)

            if expenses:
                for expense in expenses:
                    print(f"Date: {expense[0]}, Amount: {expense[1]}, Currency: {expense[2]}, Category: {expense[3]}, Description: {expense[4]}")
                return "Expenses displayed."
            else:
                return "No expenses recorded."
    except FileNotFoundError:
        return "No expense file found."


def add_event(event_date, event_name, events):
    try:
        # Parse the date and time
        event_datetime = datetime.datetime.strptime(event_date, '%Y-%m-%d %H:%M')
        events.append({"name": event_name, "date": event_date.split()[0], "time": event_date.split()[1]})
        speak(f"Event {event_name} added on {event_datetime.strftime('%B %d, %Y at %I:%M %p')}")
        print(f"Event added: {event_name} on {event_datetime}")
    except ValueError:
        speak("The date format is incorrect. Please use YYYY-MM-DD HH:MM.")

def get_current_date_time():
    current_datetime = datetime.datetime.now()
    current_date = current_datetime.strftime("%B %d, %Y")
    current_time = current_datetime.strftime("%I:%M %p")
    speak(f"Today is {current_date} and the current time is {current_time}")
    print(f"Current date: {current_date}")
    print(f"Current time: {current_time}")

def get_upcoming_events(events, current_date):
    upcoming_events = [event for event in events if event['date'] >= current_date]
    if upcoming_events:
        return upcoming_events
    else:
        return None

# Main voice-activated calendar function
def voice_activated_calendar():
    events = []  # List to store events
    speak("Welcome to the voice-activated calendar.")
    speak("What would you like to do?")
    speak("add any event")
    speak("know about upcoming event")
    speak("current date or current time")
    
    while True:
        command = listen()

        if "add event" in command:
            speak("Please say the event name.")
            event_name = listen()

            speak("Please say the event date in the format YYYY hiphen MM hiphen DD.")
            event_date = listen()

            speak("Please say the event time in 24-hour format, HH:MM.")
            event_time = listen()

            add_event(f"{event_date} {event_time}", event_name, events)
        
        elif "current date" in command or "current time" in command:
            get_current_date_time()
        
        elif "upcoming" in command and event in command:
            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
            upcoming = get_upcoming_events(events, current_date)
            if upcoming:
                speak("Here are the upcoming events.")
                for event in upcoming:
                    speak(f"{event['name']} on {event['date']} at {event['time']}")
            else:
                speak("There are no upcoming events.")
        
        elif "exit" in command or "stop" in command:
            speak("Goodbye!")
            break

        else:
            speak("I didn't understand that command. Please say 'add event', 'current date', 'upcoming events', or 'exit'.")

def add_flashcard():
    question = process_voice_input("Please say the question for the flashcard.")
    if question in ["timeout", "unknown", "service error", "error"]:
        return  # Exit if there was an error with the input
    answer = process_voice_input("Please say the answer for the flashcard.")
    if answer in ["timeout", "unknown", "service error", "error"]:
        return  # Exit if there was an error with the input
    flashcards.append((question, answer))
    engine.say("Flashcard added.")
    engine.runAndWait()

def get_recipes(ingredients):
    """Fetch recipes from the Spoonacular API based on the ingredients."""
    api_key = "200f46b8c6ae461b862345d77db700cb"
    url = f'https://api.spoonacular.com/recipes/findByIngredients?ingredients={ingredients}&apiKey={api_key}'
    response = requests.get(url)
    return response.json()


def view_score():
    global correct_answers, total_questions
    if total_questions > 0:
        score = (correct_answers / total_questions) * 100
        engine.say(f"You have answered {correct_answers} out of {total_questions} questions correctly. Your score is {score:.2f} percent.")
    else:
        engine.say("You have not answered any questions yet.")
    engine.runAndWait()

def recipe():
    # Get ingredients from the user using voice input
    ingredients = process_voice_input("Please say the ingredients you have, separated by commas.")
    if not ingredients:
        speak("No ingredients provided. Exiting.")
        return

    # Fetch recipes
    recipes = get_recipes(ingredients)

    # Check if recipes were retrieved successfully
    if isinstance(recipes, list) and recipes:
        # Announce the number of recipes found
        speak(f"I found {len(recipes)} recipes based on your ingredients.")
        # List the recipes
        for index, recipe in enumerate(recipes):
            print(f"{index + 1}. {recipe['title']}")
            print(f"   Link: {recipe['id']}")
            speak(f"{index + 1}. {recipe['title']}")
    else:
        speak("Sorry, I couldn't find any recipes. Please try again.")
        return

    # Ask the user which recipe they want to open
    recipe_number=listen_for_number("Please say the number of the recipe you want to open, or say 'exit' to quit.")
    print(recipe_number)
    if recipe_number and recipe_number.isdigit():
        recipe_number = int(recipe_number)
        if 1 <= recipe_number <= len(recipes):
            recipe_id = recipes[recipe_number - 1]['id']
            # Construct the full URL for the recipe
            recipe_url = f'https://spoonacular.com/recipes/{recipe_id}'
            webbrowser.open(recipe_url)
            speak("Opening the recipe in your web browser.")
        else:
            speak("The recipe number is out of range. Please try again.")
    elif recipe_number == "exit":
        speak("Exiting the program.")
    else:
        speak("Invalid input. Please provide a valid number or say 'exit'.")

speak("Hello BOSS ... I am your voice assistant Nova... How are you?")


r = sr.Recognizer()
translator = Translator()

def translate_text(text, dest_language='en'):
    try:
        # Translate the text
        translated = translator.translate(text, dest=dest_language)
        return translated.text
    except Exception as e:
        return f"Translation failed: {str(e)}"
    
def listen():
    with sr.Microphone() as source:
        r.energy_threshold = 10000  
        r.adjust_for_ambient_noise(source, duration=1)  
        print("Listening...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print(f"You said: {text}")
            return text.strip()
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return ""
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return ""

text = listen()

if ("what" in text and "about" in text and "you" in text ) or ("how " in text and "are" in text and "you" in text):
    speak("I am absolutely fine...")

speak("How can I help you?")
while text.lower() != "exit":
    text = listen()
    text=text.lower()
    
    if "say" in text and "hi" in text:
        speak("Hi sir ")
    
    # Repeat mode
    elif "activate" in text and "repeat" in text and "mode" in text:
        speak("Repeat mode activated. What do you want to say?")
        while True:
            text = listen()
            if text.lower() == "exit":
                speak("Exiting repeat mode.")
                break
            if text:
                speak(text)
                

    # Diary mode
    elif "activate" in text and "diary" in text and "mode" in text:
        speak("Activating diary mode.")
        speak("Enter the pass to confirm...")
        pass1 = listen()
        
        if pass1.strip() == "123":  # Strips whitespace from input
            # Get today's date
            today_date = datetime.today().strftime('%Y-%m-%d')
            path = "D:\\2nd year\\python_assistant\\new1.txt"
            # Ensure the directory exists
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # Write today's date to the file
            try:
                with open(path, 'w') as file:
                    file.write(f"{today_date}   ")  # Add a newline for clarity
                    print(f"Date written to file: {today_date}")  # Debugging output
                    
                speak("What would you like to record for today?")
                content = listen()
                
                if content:  # Only write if content is not empty
                    with open(path, 'a') as file:
                        file.write(f"{content}\n")
                        print(f"Content written to file: {content}")  # Debugging output
                    speak("Your entry has been saved.")
                else:
                    speak("No content recorded for today.")                    
            except Exception as e:
                print(f"Error writing to file: {e}")
                speak("An error occurred while saving your entry.")
        else:
            speak("Wrong password... Exiting from diary mode.")



    # Reader mode
    elif "activate" in text and "reader" in text and "mode" in text:
        speak("Activating reader mode.")
        speak("Say the date you want to read the record for.")
        path = "D:\\2nd year\\python_assistant\\new1.txt"
        date_input = listen()  # Listen for the date input from the user

        try:
            # Convert spoken date to the desired format (YYYY-MM-DD)
            date_obj = datetime.strptime(date_input, '%d %m %Y')
            formatted_date = date_obj.strftime('%Y-%m-%d')
            
            # Open the file to read
            with open(path, 'r') as file:
                found = False
                for line in file:
                # Check if the current line matches the formatted date
                    if formatted_date in line:  
                        speak("Date matched.")
                        speak(line) 
                        print(line) # Read the line corresponding to the date
                        found = True
                        read_record = True  # Start reading subsequent lines
                        continue  # M
                    
                if not found:
                    speak("No records found for the specified date.")
        
        except FileNotFoundError:
            speak("The diary file does not exist. Please check the file path.")
        except ValueError:
            speak("The date format you provided is incorrect. Please use the format 'DD MM YYYY'.")
        except Exception as e:
            print(f"An error occurred: {e}")
            speak("An error occurred while reading the file.")

    elif "activate" in text and "translator" in text:
        speak("Activating Translator")
        speak("which language you need to tranlate")
        language=listen()
        speak("You can translate now")
        text=listen()
        translation = translate_text(text.strip(),dest_language=language.strip())
        print(f"Translated to {language.strip()}: {translation}")
        speak(translation)
    elif "tell" in text and "joke" in text:
        joke = pyjokes.get_joke()
        print("Here's a joke for you:",joke)
        speak(joke)
    elif "tell" in text and "quotes" in text or "quote" in text:
        response = requests.get("https://zenquotes.io/api/random")
        if response.status_code == 200:
            data = response.json()
            quote = f"{data[0]['q']} - {data[0]['a']}"
            print("The Quote is :",quote)
            speak( quote)
        else:
            print("Failed to fetch a quote.")
    elif "tell" in text and "weather" in text and "forecast" in text:
        speak("Tell the city name to find the forecast: ")
        city=listen()
        city=city.lower()
        api_key = "35df73c218d549d6aa3165644241810"
        base_url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
        response = requests.get(base_url)
        if response.status_code == 200:
            data = response.json()
            temperature = data['current']['temp_c']
            weather_description = data['current']['condition']['text']
            humidity = data['current']['humidity']
            weather_info = (
                f"The current temperature in {city} is {temperature} degrees Celsius. "
                f"The weather is {weather_description} with a humidity of {humidity} percent."
            )
            print(weather_info)
            speak(weather_info)
        else:
            error_message = "Failed to fetch weather information. Please check the city name or try again later."
            speak(error_message)
    elif "tell" in text and "news" in text:
        api_key = "165011e733ac47439f2e7696b0f1d379"  
        speak("Enter the news category (business, entertainment, general, health, science, sports, technology,education): ")
        category = listen()
        url = f'https://newsapi.org/v2/everything?q={category}&apiKey={api_key}'
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json()['articles']
            news_brief = ""
            for article in articles[:3]:  # Get top 5 articles
                title = article['title']
                description = article['description']
                news_brief += f"Title: {title}. Description: {description}\n"
            print(news_brief)
            speak(news_brief)
        else:
            error_message = "Failed to fetch news articles."
            print(error_message)
            speak(error_message)
    elif "activate" in text and "flash card" in text and "mode" in text:
        calibrate_microphone()  # Call the calibration function
        engine.say("Welcome to the flashcard assistant. You can add flashcards or start the quiz. Say 'add' to add a flashcard, or 'quiz' to start the quiz.")
        engine.runAndWait()

        while True:
            command = process_voice_input("What would you like to do? Say 'add' to add a flashcard or 'quiz' to start the quiz.")
            
            if command in ["add", "add flashcards", "adding flashcards", "add flash card", "add flash cards"]:
                print("Adding flashcards...")
                add_flashcard()
            elif command in ["quiz", "start quiz"]:
                print("Starting quiz...")
                quiz_mode()
            elif command in ["exit", "quit"]:
                engine.say("Goodbye!")
                engine.runAndWait()
                break
            else:
                print("Unrecognized command. Please try again.")
                engine.say("I did not understand that. Please say 'add' or 'quiz'.")
                engine.runAndWait()
    elif "tell" in text and "recipe" in text :
        speak("Welcome to the recipe assistant.")
        recipe()
        speak("Goodbye!") 
    elif "activate" in text and "calendar" in text:
         voice_activated_calendar()
    elif "activate" in text and "calculator" in text:
        engine.say("Welcome to the voice-activated calculator. Say 'exit' to quit.")
        engine.runAndWait()

        while True:
            # Get voice input
            input_text = process_voice_input_calculator()

            # Check if the user wants to exit
            if "exit" in input_text:
                engine.say("Goodbye!")
                engine.runAndWait()
                break

            # Calculate the result based on voice input
            response = evaluate_expression(input_text)
            
            # Output the result
            engine.say(response)
            engine.runAndWait()
            print(response)
    elif "activate" in text and "tracker" in text:
        try:
            with open(CSV_FILE, mode='x', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Date", "Amount", "Currency", "Category", "Description"])
        except FileExistsError:
            pass

        # Voice assistant for adding expenses
        engine.say("Welcome to your voice-activated expense tracker. Say 'exit' to quit or 'show expenses' to view your expenses.")
        engine.runAndWait()

        while True:
            # Get voice input
            input_text = process_expense_input()

            # Check if the user wants to exit
            if "exit" in input_text:
                engine.say("Goodbye!")
                engine.runAndWait()
                break

            # Check if the user wants to show expenses
            elif "show expenses" in input_text:
                response = show_expenses()
                engine.say(response)
                engine.runAndWait()

            # Parse and add the expense
            elif "add expenses" in input_text:
                print("Into Adding Expenses")
                details = parse_expense_details(input_text)
                if details:
                    amount, currency, category, description = details
                    add_expense(amount, currency, category, description)
                    response = f"Added {amount} {currency} for {category}. Description: {description}."
                else:
                    response = "I could not understand your expense. Please try again."

                # Speak the response
                engine.say(response)
                engine.runAndWait()

    elif "change" in text and "voice" in text:
        speak("Sure ..I will change my voice")
        print("Process undergoing....")
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        speak("Voice sucessfully changed")
    elif "what" in text and "your" in text and "boss name" in text:
        speak("my boss name is kavin")
    elif "what" in text and "your" in text and "name" in text:
        speak("My name is Nova....A voice Assistant")
    elif "google" in text:
        speak("searching in Google")
        searchGoogle(text)
    elif "youtube" in text:
        speak("Searching in youtube")
        searchYoutube(text)
    elif "wikipedia" in text:
        speak("Searching in Wikipedia")
        searchWikipedia(text)
    else:
        if text=='':
            print("Try again")
        else:          
            last_response = chat_with_bot(text)  # Start the chat with the provided initial message
            #print("Last response from chatbot:", last_response) 