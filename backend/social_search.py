from urllib.parse import quote_plus


def search_social_media(name, location=""):
    query = f"{name} {location}".strip()
    encoded = quote_plus(query)

    return [
        {
            "platform": "Google",
            "url": f"https://www.google.com/search?q={encoded}",
            "description": "Public web results"
        },
        {
            "platform": "Facebook",
            "url": f"https://www.google.com/search?q=site%3Afacebook.com+{encoded}",
            "description": "Public Facebook results"
        },
        {
            "platform": "Instagram",
            "url": f"https://www.google.com/search?q=site%3Ainstagram.com+{encoded}",
            "description": "Public Instagram results"
        },
        {
            "platform": "X",
            "url": f"https://www.google.com/search?q=site%3Ax.com+{encoded}",
            "description": "Public X results"
        },
        {
            "platform": "LinkedIn",
            "url": f"https://www.google.com/search?q=site%3Alinkedin.com+{encoded}",
            "description": "Public LinkedIn results"
        },
        {
            "platform": "YouTube",
            "url": f"https://www.google.com/search?q=site%3Ayoutube.com+{encoded}",
            "description": "Public YouTube results"
        },
        {
            "platform": "TikTok",
            "url": f"https://www.google.com/search?q=site%3Atiktok.com+{encoded}",
            "description": "Public TikTok results"
        },
        {
            "platform": "Reddit",
            "url": f"https://www.google.com/search?q=site%3Areddit.com+{encoded}",
            "description": "Public Reddit results"
        }
    ]