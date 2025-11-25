import os
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ConfigDict
from browser_use_sdk import AsyncBrowserUse

# --- CONFIGURATION ---
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
if "BROWSER_USE_API_KEY" not in os.environ:
    raise ValueError("BROWSER_USE_API_KEY environment variable not set")

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
    client = AsyncBrowserUse(api_key=os.environ["BROWSER_USE_API_KEY"])
    
    try:
        task = await client.tasks.create_task(
            task="Go to google.com, type 'AAPL stock price' into the search bar, and tell me the current price.",
            llm=llm,
        )
        print(">> Agent Initialized. Starting task...")
        result = await task.complete()
        print("\n>> Task Complete!")
        print(result.output)
        
    except Exception as e:
        print(f"\n>> CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # The new SDK handles browser cleanup automatically, so no need to close the browser manually.
        pass

if __name__ == "__main__":
    asyncio.run(main())