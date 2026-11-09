from app.gateway import create_chat_completion


def chat_response(user_message):

    response = create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": """
You are a helpful AI assistant.
Answer concisely.
"""
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    return response.choices[0].message.content