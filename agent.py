import os
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ConfigDict

# --- UPDATED IMPORTS FOR DEV VERSION ---
# The class locations changed in the latest update. 
# We now import directly from the root package.
from browser_use import Agent, Browser, BrowserConfig

# --- CONFIGURATION ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyC5cncyqaWgE_pvogTR4Soihcbv7Cc6et8"

# --- THE FIX: PERMISSIVE WRAPPER v3 (Must Keep!) ---
class PermissiveGemini(ChatGoogleGenerativeAI):
    model_config = ConfigDict(extra="allow")
    provider: str = "google"

    @property
    def model_name(self):
        return self.model

async def main():
    # We use gemini-1.5-flash as it is faster and often stricter with JSON
    llm = PermissiveGemini(model="gemini-1.5-flash")

    # Initialize Browser
    # We set headless=False so you can watch the agent work
    browser = Browser(
        config=BrowserConfig(
            headless=False,
        )
    )

    # Agent Definition
    agent = Agent(
        task="Go to google.com, type 'AAPL stock price' into the search bar, and tell me the current price.",
        llm=llm,
        browser=browser,
    )

    print(">> Agent Initialized. Starting task...")
    
    try:
        # Run the agent
        history = await agent.run()
        
        print("\n>> Task Complete!")
        print(history.final_result())
        
    except Exception as e:
        print(f"\n>> CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always close the browser cleanly
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())