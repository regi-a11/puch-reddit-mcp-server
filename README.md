# 🏆 Hackathon Submission: Reddit MCP Server

This project is a **Model Context Protocol (MCP)** server built for the Puch AI Hackathon. It connects Puch AI to the Reddit platform, acting as a powerful read-only assistant for fetching and analyzing public Reddit data.

The server is built on the `FastMCP` framework, ensuring seamless integration and robust performance.

---

## ✨ How to Use Our MCP Server

Once connected, you can interact with our server using natural language. The AI will intelligently route your request to the appropriate tool.

Here are some example commands and the feedback you can expect:

### 1. Get Top Posts from a Subreddit

This tool fetches the top posts from any public subreddit, complete with AI-driven engagement analysis.

* **Command:** `Show me the top posts from r/Python this week`
* **Expected Feedback:**
    ```
    🏆 **Top 5 Posts from r/Python (week):**

    ---

    • Title: New to Python? Start here!
    • Type: Text Post
    • Content: (Brief post content)
    ...
    📈 Engagement Analysis:
      - Highly successful post with strong community approval.
      - Generated significant discussion.

    ---
    (More posts will follow...)
    ```

### 2. Get User Information

This tool provides a detailed summary of any Reddit user's profile, including their karma and a basic activity overview.

* **Command:** `Get Reddit user info for u/AutoModerator`
* **Expected Feedback:**
    ```
    👤 **Reddit User Info: u/AutoModerator**

    • Created: 2011-04-20 20:53:14 UTC
    • Karma (Comment): 1,234,567
    • Karma (Link): 890
    • Total Karma: 1,235,457
    ...
    ```

### 3. Get Subreddit Information & Stats

This provides a detailed overview of a subreddit, including its title, subscriber count, and active user count.

* **Command:** `What are the stats for the subreddit 'machinelearning'?`
* **Expected Feedback:**
    ```
    📊 **Subreddit Stats: r/machinelearning**

    • Full Title: Machine Learning
    • Public Description: Welcome to r/MachineLearning...
    • Total Subscribers: 3,456,789
    • Active Users (now): 12,345
    ...
    ```

### 4. Get Trending Subreddits

This tool is useful for discovering what's popular on Reddit right now.

* **Command:** `What are the trending subreddits?`
* **Expected Feedback:**
    ```
    🔥 **Currently Trending Subreddits:**

    ---

    1. **r/cats**
       • Subscribers: 5,678,901
       • Description: All things cats...
    2. **r/technology**
       • Subscribers: 9,876,543
       • Description: For the latest in technology news...
    ...
    ```

### 5. Analyze a Specific Post

You can get detailed information about any Reddit post just by providing its URL or ID.

* **Command:** `Analyze the Reddit post at https://www.reddit.com/r/pics/comments/123456/my_awesome_picture/`
* **Expected Feedback:**
    ```
    • Title: My awesome picture
    • Type: Link Post
    • Content: (URL)
    • Author: u/example_user
    ...
    📈 Engagement Analysis:
      - Well-received post with good engagement.
    ...
    ```

---

## 💻 Quick Setup Guide

This guide is for developers who want to set up and run this project locally.

### Step 1: Clone and Install
1.  Clone the repository and navigate to the project root.
2.  Install dependencies using `uv` (as defined in `pyproject.toml`):
    ```bash
    # From the project root (where pyproject.toml is)
    uv venv
    uv sync
    source .venv/bin/activate
    ```

### Step 2: Configure Credentials
1.  Create a `.env` file in the `mcp-bearer-token/` directory.
2.  Add your Puch AI and Reddit API credentials:
    ```env
    AUTH_TOKEN=your_secret_token
    MY_NUMBER=your_whatsapp_number
    REDDIT_USER_AGENT=PuchAIRedditMCP-ReadOnly/1.0
    ```

### Step 3: Run and Deploy
1.  Run the server from the `mcp-bearer-token/` directory:
    ```bash
    python mcp_starter.py
    ```
2.  Deploy for 24/7 access using a service like **Railway** by connecting your GitHub repo and configuring your environment variables in their dashboard.

---

**Happy hacking! 🚀** Use `#BuildWithPuch` to share your project.
