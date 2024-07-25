from openai import OpenAI
import json

client = OpenAI()
model="gpt-4o-mini"
model="gpt-4o"
# model="gpt-3.5-turbo-0125"

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    print("get +++current")
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def get_future_weather(location, unit="fahrenheit"):
    print("get +++future")
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "110", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "172", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "122", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def run_conversation():
    # Step 1: send the conversation and available functions to the model
    # messages_list = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
    messages_list = [{"role": "user", "content": "What was the weather  like 100 years ago year  in San Francisco, Tokyo, and Paris?"}]
    messages_list = [{"role": "user", "content": "What will the weather be  like next year in San Francisco, Tokyo, and Paris?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_future_weather",
                "description": "Get the future weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        },
    ]

    available_functions = {
        "get_current_weather": get_current_weather,
        "get_future_weather": get_future_weather,
    }

    response1 = client.chat.completions.create(
        model=model,
        messages=messages_list,
        tools=tools,
        seed=1,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response1.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        messages_list.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_ptr = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            location1=function_args.get("location")
            unit1=function_args.get("unit")
            print("location", location1)
            function_response = function_ptr(
                location1,
                unit1,
            )
            messages_list.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        response2 = client.chat.completions.create(
            model=model,
            messages=messages_list,
            seed=1,
            temperature=1.2
        )  # get a new response from the model where it can see the function response
        for msg in messages_list:
            print("===")
            print(msg)
        return response2
    else:
        return response1
    
resp=run_conversation()
print("final =========\n",resp.choices[0].message.content)
