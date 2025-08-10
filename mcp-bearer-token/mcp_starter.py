import asyncio
import os
import functools
import logging
import time
from datetime import datetime
from typing import Annotated, Optional, List, Dict, Any, Callable, TypeVar, cast

# External libraries
from dotenv import load_dotenv
import praw  # Python Reddit API Wrapper
from pydantic import BaseModel, Field, HttpUrl

# FastMCP and MCP-specific imports
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, INVALID_PARAMS, INTERNAL_ERROR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variable for decorators (though not strictly needed now for write access)
F = TypeVar("F", bound=Callable[..., Any])

# --- Load environment variables ---
load_dotenv()

# Puch AI authentication token
TOKEN = os.environ.get("AUTH_TOKEN")
# Your WhatsApp number (required by Puch AI for validation)
MY_NUMBER = os.environ.get("MY_NUMBER")

# Reddit API credentials (Client ID and Secret are still needed for read-only access)
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
# Username and Password are NOT used for read-only operations, and are now optional.
# They will be ignored by the RedditClientManager in this read-only setup.
REDDIT_USERNAME = os.environ.get("REDDIT_USERNAME")
REDDIT_PASSWORD = os.environ.get("REDDIT_PASSWORD")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "PuchAIRedditMCP-ReadOnly/1.0 (by u/YourRedditUsername)")


# Assert that essential variables are set
assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert REDDIT_CLIENT_ID is not None, "Please set REDDIT_CLIENT_ID in your .env file for Reddit access."
assert REDDIT_CLIENT_SECRET is not None, "Please set REDDIT_CLIENT_SECRET in your .env file for Reddit access."


# --- Puch AI Authentication Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    """
    Custom bearer token provider for Puch AI authentication.
    It verifies the AUTH_TOKEN provided by the server owner.
    """
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        """
        Loads the access token if it matches the configured AUTH_TOKEN.
        """
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client", # A generic client ID for Puch AI
                scopes=["*"],            # Full access to tools
                expires_at=None,
            )
        logger.warning(f"Invalid bearer token provided: {token}")
        return None

# --- Reddit Client Manager (FORCED READ-ONLY) ---
class RedditClientManager:
    """
    Manages the PRAW (Python Reddit API Wrapper) client.
    FORCED to initialize in read-only mode to prevent accidental write operations.
    """
    _instance = None
    _client: Optional[praw.Reddit] = None
    _is_read_only: bool = True # Always True in this configuration

    def __new__(cls) -> "RedditClientManager":
        """
        Ensures a single instance of RedditClientManager (singleton).
        Initializes the PRAW client upon first instantiation.
        """
        if cls._instance is None:
            cls._instance = super(RedditClientManager, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self) -> None:
        """
        Initializes the PRAW client in read-only mode using client ID and secret.
        Username/password are ignored for security.
        """
        try:
            logger.info("Initializing Reddit client in FORCED read-only mode.")
            self._client = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
                check_for_updates=False, # Disable PRAW's update checker
                read_only=True, # Explicitly force read-only
            )
            # Test read-only access by fetching a popular subreddit
            self._client.subreddit("popular").hot(limit=1)
            self._is_read_only = True # Confirm read-only state

        except Exception as e:
            logger.error(f"FATAL: Error initializing Reddit client in read-only mode: {e}. All Reddit tools may fail.")
            self._client = None
            self._is_read_only = True # Ensure it's marked as read-only even on failure

    @property
    def client(self) -> Optional[praw.Reddit]:
        """Returns the PRAW client instance."""
        return self._client

    @property
    def is_read_only(self) -> bool:
        """Checks if the PRAW client is in read-only mode (always True in this setup)."""
        return self._is_read_only

# Initialize the Reddit client manager globally
reddit_manager = RedditClientManager()

# --- Helper Functions for Formatting and Analysis (Unchanged) ---

def _format_timestamp(timestamp: float) -> str:
    """Converts a Unix timestamp to a human-readable UTC string."""
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return "N/A"

def _analyze_post_engagement(score: int, ratio: float, num_comments: int) -> str:
    """Provides AI-driven insights on Reddit post engagement."""
    insights = []
    if score > 1000 and ratio > 0.95:
        insights.append("Highly successful post with strong community approval.")
    elif score > 100 and ratio > 0.8:
        insights.append("Well-received post with good engagement.")
    elif ratio < 0.5:
        insights.append("Controversial post that sparked debate.")
    if num_comments > 100:
        insights.append("Generated significant discussion.")
    elif num_comments > score * 0.5:
        insights.append("Highly discussable content with active comment section.")
    elif num_comments == 0:
        insights.append("Yet to receive community interaction.")
    return "\n  - " + "\n  - ".join(insights) if insights else "No specific engagement patterns detected."

def _get_best_engagement_time(created_utc: float, score: int) -> str:
    """Analyzes a post's creation time and suggests optimal posting times."""
    post_hour = datetime.fromtimestamp(created_utc).hour
    if 14 <= post_hour <= 18:
        return "Posted during peak engagement hours (2 PM - 6 PM UTC), good timing!"
    elif 23 <= post_hour or post_hour <= 5:
        return "Consider posting during more active hours (morning to evening UTC)."
    else:
        return "Posted during moderate activity hours, timing could be optimized."

def _format_post_output(post: praw.models.Submission) -> str:
    """Formats Reddit submission (post) information for user display."""
    content_type = "Text Post" if post.is_self else "Link Post"
    content = post.selftext if post.is_self else post.url
    
    flags = []
    if post.over_18: flags.append("NSFW")
    if hasattr(post, "spoiler") and post.spoiler: flags.append("Spoiler")
    if post.edited: flags.append("Edited")

    image_url_section = f"\n• Image URL: {post.url}" if not post.is_self else ""

    return (
        f"• Title: {post.title}\n"
        f"• Type: {content_type}\n"
        f"• Content: {content}\n"
        f"• Author: u/{str(post.author)}\n"
        f"• Subreddit: r/{str(post.subreddit)}{image_url_section}\n"
        f"• Stats:\n"
        f"  - Score: {post.score:,}\n"
        f"  - Upvote Ratio: {post.upvote_ratio * 100:.1f}%\n"
        f"  - Comments: {post.num_comments:,}\n"
        f"• Metadata:\n"
        f"  - Posted: {_format_timestamp(post.created_utc)}\n"
        f"  - Flags: {', '.join(flags) if flags else 'None'}\n"
        f"  - Flair: {post.link_flair_text or 'None'}\n"
        f"• Links:\n"
        f"  - Full Post: https://reddit.com{post.permalink}\n"
        f"  - Short Link: https://redd.it/{post.id}\n\n"
        f"📈 Engagement Analysis:\n{_analyze_post_engagement(post.score, post.upvote_ratio, post.num_comments)}\n\n"
        f"🎯 Best Time to Engage:\n  - {_get_best_engagement_time(post.created_utc, post.score)}"
    )

def _extract_reddit_id(reddit_id: str) -> str:
    """Extracts the base Reddit ID from a URL or ID string."""
    if not reddit_id:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Empty ID provided."))
    if "/" in reddit_id:
        parts = [p for p in reddit_id.split("/") if p]
        reddit_id = parts[-1]
    return reddit_id

# No _format_comment_output needed as write ops are removed.

# --- MCP Server Setup ---
mcp = FastMCP(
    "Reddit Read-Only MCP Server", # Name of your server
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    """
    Validates the MCP server connection and returns the owner's phone number.
    This tool is required by Puch AI for initial server validation.
    """
    return MY_NUMBER

# --- Tool: get_user_info ---
@mcp.tool(
    description="Get information about a Reddit user, including karma, creation date, and various flags. "
                "Provides AI-driven analysis of engagement patterns."
)
async def get_user_info(
    username: Annotated[str, Field(description="The username of the Reddit user (with or without 'u/' prefix).")]
) -> TextContent:
    """
    Fetches detailed information about a specified Reddit user.
    """
    manager = reddit_manager
    if not manager.client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="Reddit client not initialized. Cannot fetch user info."))

    if not username or not isinstance(username, str) or username.startswith((" ", "/")):
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Invalid username provided. Please provide a valid Reddit username."))

    clean_username = username[2:] if username.startswith("u/") else username

    try:
        logger.info(f"Getting info for u/{clean_username}")
        user = manager.client.redditor(clean_username)
        # Force fetch user data to verify it exists
        _ = user.created_utc

        # Prepare formatted output string
        output = (
            f"👤 **Reddit User Info: u/{user.name}**\n\n"
            f"• Created: {_format_timestamp(user.created_utc)}\n"
            f"• Karma (Comment): {user.comment_karma:,}\n"
            f"• Karma (Link): {user.link_karma:,}\n"
            f"• Total Karma: {getattr(user, 'total_karma', user.comment_karma + user.link_karma):,}\n"
            f"• Verified Email: {'Yes' if getattr(user, 'has_verified_email', False) else 'No'}\n"
            f"• Is Moderator: {'Yes' if getattr(user, 'is_mod', False) else 'No'}\n"
            f"• Is Gold: {'Yes' if getattr(user, 'is_gold', False) else 'No'}\n"
            f"• Over 18: {'Yes' if getattr(user, 'over_18', False) else 'No'}\n"
            f"• Suspended: {'Yes' if getattr(user, 'is_suspended', False) else 'No'}"
        )

        if hasattr(user, 'subreddit') and user.subreddit and getattr(user.subreddit, 'display_name', None):
            output += (
                f"\n• Profile Subreddit: r/{user.subreddit.display_name}\n"
                f"  - Title: {getattr(user.subreddit, 'title', 'N/A')}\n"
                f"  - Description: {getattr(user.subreddit, 'public_description', 'N/A')}\n"
                f"  - Subscribers: {getattr(user.subreddit, 'subscribers', 0):,}"
            )

        return TextContent(type="text", text=output)

    except Exception as e:
        logger.error(f"Error getting user info for u/{clean_username}: {e}")
        if "USER_DOESNT_EXIST" in str(e) or "NOT_FOUND" in str(e).upper():
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"User u/{clean_username} not found.")) from e
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to retrieve user information: {e}")) from e

# --- Tool: get_top_posts ---
@mcp.tool(
    description="Get top posts from a specific subreddit, filtered by time period. "
                "Provides engagement analysis and optimal posting time insights for each post."
)
async def get_top_posts(
    subreddit: Annotated[str, Field(description="Name of the subreddit (with or without 'r/' prefix).")],
    time_filter: Annotated[str, Field(description="Time period to filter posts (e.g., 'day', 'week', 'month', 'year', 'all').", pattern="^(hour|day|week|month|year|all)$")] = "week",
    limit: Annotated[int, Field(description="Number of posts to fetch (1-10, default is 5).", ge=1, le=10)] = 5,
) -> TextContent:
    """
    Fetches and formats a list of top posts from a specified subreddit.
    """
    manager = reddit_manager
    if not manager.client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="Reddit client not initialized. Cannot fetch posts."))

    if not subreddit or not isinstance(subreddit, str):
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Subreddit name is required."))

    clean_subreddit = subreddit[2:] if subreddit.startswith("r/") else subreddit

    try:
        logger.info(f"Getting top {limit} posts from r/{clean_subreddit} (time_filter={time_filter})")
        sub = manager.client.subreddit(clean_subreddit)
        _ = sub.display_name # Verify subreddit exists

        posts = list(sub.top(time_filter=time_filter, limit=limit))

        if not posts:
            return TextContent(type="text", text=f"No top posts found for r/{clean_subreddit} in the last {time_filter}.")

        formatted_posts = "\n\n---\n\n".join([_format_post_output(p) for p in posts])

        return TextContent(type="text", text=
            f"🏆 **Top {len(posts)} Posts from r/{clean_subreddit} ({time_filter}):**\n\n"
            f"{formatted_posts}"
        )

    except Exception as e:
        logger.error(f"Error getting top posts from r/{clean_subreddit}: {e}")
        if "private" in str(e).lower() or "banned" in str(e).lower() or "not found" in str(e).lower():
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Subreddit r/{clean_subreddit} is private, banned, or not found.")) from e
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to get top posts: {e}")) from e

# --- Tool: get_subreddit_info ---
@mcp.tool(
    description="Get general information about a Reddit subreddit, including its title, description, and subscriber count."
)
async def get_subreddit_info(
    subreddit_name: Annotated[str, Field(description="Name of the subreddit (with or without 'r/' prefix).")],
) -> TextContent:
    """
    Fetches and formats general information about a specified Reddit subreddit.
    """
    manager = reddit_manager
    if not manager.client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="Reddit client not initialized. Cannot fetch subreddit info."))

    if not subreddit_name or not isinstance(subreddit_name, str):
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Subreddit name is required."))

    clean_name = subreddit_name[2:] if subreddit_name.startswith("r/") else subreddit_name

    try:
        logger.info(f"Getting info for r/{clean_name}")
        subreddit = manager.client.subreddit(clean_name)
        _ = subreddit.display_name # Verify subreddit exists

        output = (
            f"ℹ️ **Subreddit Info: r/{subreddit.display_name}**\n\n"
            f"• Title: {subreddit.title}\n"
            f"• Description: {subreddit.public_description or 'No public description.'}\n"
            f"• Subscribers: {subreddit.subscribers:,}\n"
            f"• Created: {_format_timestamp(subreddit.created_utc)}\n"
            f"• NSFW: {'Yes' if subreddit.over18 else 'No'}\n"
            f"• Type: {getattr(subreddit, 'subreddit_type', 'N/A').capitalize()}\n"
            f"• Active Users: {getattr(subreddit, 'active_user_count', 'N/A'):,}\n"
            f"• URL: {subreddit.url}"
        )
        return TextContent(type="text", text=output)

    except Exception as e:
        logger.error(f"Error getting info for r/{clean_name}: {e}")
        if "private" in str(e).lower() or "banned" in str(e).lower() or "not found" in str(e).lower():
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Subreddit r/{clean_name} is private, banned, or not found.")) from e
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to get subreddit info: {e}")) from e

# --- Tool: get_trending_subreddits ---
@mcp.tool(
    description="Get a list of currently trending subreddits, showing their names, subscriber counts, and public descriptions."
)
async def get_trending_subreddits(
    limit: Annotated[int, Field(description="Maximum number of trending subreddits to return (1-10, default is 5).", ge=1, le=10)] = 5,
) -> TextContent:
    """
    Fetches and formats a list of currently trending subreddits.
    """
    manager = reddit_manager
    if not manager.client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="Reddit client not initialized. Cannot fetch trending subreddits."))

    try:
        logger.info(f"Getting top {limit} trending subreddits")
        popular_subreddits = manager.client.subreddits.popular(limit=limit)

        trending_list = []
        for i, sub in enumerate(popular_subreddits):
            trending_list.append(
                f"{i+1}. **r/{sub.display_name}**\n"
                f"   • Subscribers: {sub.subscribers:,}\n"
                f"   • Description: {sub.public_description or 'No description.'}\n"
                f"   • NSFW: {'Yes' if sub.over18 else 'No'}\n"
                f"   • URL: {sub.url}"
            )
        
        if not trending_list:
            return TextContent(type="text", text="No trending subreddits found at the moment.")

        return TextContent(type="text", text=
            f"🔥 **Currently Trending Subreddits:**\n\n"
            + "\n\n---\n\n".join(trending_list)
        )

    except Exception as e:
        logger.error(f"Error getting trending subreddits: {e}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to get trending subreddits: {e}")) from e

# --- Tool: get_subreddit_stats ---
@mcp.tool(
    description="Get detailed statistics and features of a subreddit, including rules, moderator count, and enabled features."
)
async def get_subreddit_stats(
    subreddit: Annotated[str, Field(description="Name of the subreddit (with or without 'r/' prefix).")],
) -> TextContent:
    """
    Fetches and formats detailed statistics and features for a specified subreddit.
    """
    manager = reddit_manager
    if not manager.client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="Reddit client not initialized. Cannot fetch subreddit stats."))

    if not subreddit or not isinstance(subreddit, str):
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Subreddit name is required."))

    clean_name = subreddit[2:] if subreddit.startswith("r/") else subreddit

    try:
        logger.info(f"Getting stats for r/{clean_name}")
        sub = manager.client.subreddit(clean_name)
        sub._fetch() # Force fetch all attributes

        mod_count = "N/A"
        try:
            if hasattr(sub, "moderator"):
                mod_count = len(list(sub.moderator()))
        except Exception:
            logger.debug(f"Could not fetch moderator count for r/{clean_name}")

        rules_list = []
        try:
            if hasattr(sub, "rules"):
                for rule in sub.rules():
                    rules_list.append(f"  - {rule.short_name}: {rule.description}")
        except Exception:
            logger.debug(f"Could not fetch rules for r/{clean_name}")
        rules_str = "\n".join(rules_list) if rules_list else "  - No rules available."

        features = {
            "Wiki Enabled": getattr(sub, "wikienabled", False),
            "Spoilers Enabled": getattr(sub, "spoilers_enabled", False),
            "Polls Allowed": getattr(sub, "allow_polls", False),
            "Images Allowed": getattr(sub, "allow_images", False),
            "Videos Allowed": getattr(sub, "allow_videos", False),
            "Crossposts Allowed": getattr(sub, "allow_crossposts", True),
            "Chat Post Creation": getattr(sub, "allow_chat_post_creation", False),
        }
        features_str = "\n".join([f"  - {k}: {'Yes' if v else 'No'}" for k, v in features.items()])


        output = (
            f"📊 **Subreddit Stats: r/{sub.display_name}**\n\n"
            f"• Full Title: {sub.title}\n"
            f"• Public Description: {sub.public_description or 'N/A'}\n"
            f"• Total Subscribers: {sub.subscribers:,}\n"
            f"• Active Users (now): {getattr(sub, 'active_user_count', 'N/A'):,}\n"
            f"• Creation Date: {_format_timestamp(sub.created_utc)}\n"
            f"• NSFW: {'Yes' if sub.over18 else 'No'}\n"
            f"• Submission Type: {getattr(sub, 'submission_type', 'N/A').capitalize()}\n"
            f"• Quarantined: {'Yes' if getattr(sub, 'quarantine', False) else 'No'}\n"
            f"• URL: {sub.url}\n\n"
            f"👮 **Moderation & Rules**\n"
            f"• Moderators: {mod_count}\n"
            f"• Rules:\n{rules_str}\n\n"
            f"✨ **Features**\n{features_str}"
        )
        return TextContent(type="text", text=output)

    except Exception as e:
        logger.error(f"Error getting stats for r/{clean_name}: {e}")
        if "private" in str(e).lower() or "banned" in str(e).lower() or "not found" in str(e).lower():
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Subreddit r/{clean_name} is private, banned, or not found.")) from e
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to get subreddit stats: {e}")) from e

# --- Tool: get_submission_by_url ---
@mcp.tool(
    description="Get detailed information about a Reddit post (submission) by its full URL."
)
async def get_submission_by_url(
    url: Annotated[HttpUrl, Field(description="The full URL of the Reddit submission to retrieve.")]
) -> TextContent:
    """
    Fetches and formats detailed information for a Reddit submission given its URL.
    """
    manager = reddit_manager
    if not manager.client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="Reddit client not initialized. Cannot fetch submission by URL."))

    try:
        logger.info(f"Getting submission from URL: {url}")
        submission = manager.client.submission(url=str(url))
        _ = submission.title # Verify submission exists

        return TextContent(type="text", text=_format_post_output(submission))

    except Exception as e:
        logger.error(f"Error getting submission by URL {url}: {e}")
        if "404" in str(e) or "not found" in str(e).lower():
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Submission not found at URL: {url}")) from e
        if "403" in str(e) or "forbidden" in str(e).lower():
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Not authorized to access submission at URL: {url}")) from e
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to get submission by URL: {e}")) from e

# --- Tool: get_submission_by_id ---
@mcp.tool(
    description="Get detailed information about a Reddit post (submission) by its ID."
)
async def get_submission_by_id(
    submission_id: Annotated[str, Field(description="The ID of the Reddit submission to retrieve.")]
) -> TextContent:
    """
    Fetches and formats detailed information for a Reddit submission given its ID.
    """
    manager = reddit_manager
    if not manager.client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="Reddit client not initialized. Cannot fetch submission by ID."))

    if not submission_id:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Submission ID is required."))

    try:
        clean_submission_id = _extract_reddit_id(submission_id)
        logger.info(f"Getting submission with ID: {clean_submission_id}")
        submission = manager.client.submission(id=clean_submission_id)
        _ = submission.title # Verify submission exists

        return TextContent(type="text", text=_format_post_output(submission))

    except Exception as e:
        logger.error(f"Error getting submission by ID {submission_id}: {e}")
        if "404" in str(e) or "not found" in str(e).lower():
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Submission with ID '{clean_submission_id}' not found.")) from e
        if "403" in str(e) or "forbidden" in str(e).lower():
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Not authorized to access submission with ID '{clean_submission_id}'.")) from e
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to get submission by ID: {e}")) from e


# --- Run MCP Server ---
async def main():
    print("🚀 Starting Reddit Read-Only MCP server on http://0.0.0.0:8086")
    print("Initializing Reddit client in read-only mode...")
    # Ensure Reddit client is initialized before running the server
    _ = reddit_manager.client
    if not reddit_manager.client:
        print("WARNING: Reddit client failed to initialize. Please check REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in your .env file.")
    else:
        print("Reddit client initialized successfully in read-only mode.")

    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())

