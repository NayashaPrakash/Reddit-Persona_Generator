import json
import time
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import google.generativeai as genai


@dataclass
class RedditContent:
    """Data class for Reddit content (posts/comments)"""
    id: str
    title: str
    body: str
    subreddit: str
    score: int
    created_utc: float
    permalink: str
    content_type: str  # 'post' or 'comment'


@dataclass
class UserPersona:
    """Data class for user persona"""
    username: str
    demographics: Dict[str, str]
    interests: List[str]
    personality_traits: List[str]
    communication_style: str
    activity_patterns: Dict[str, str]
    goals_motivations: List[str]
    pain_points: List[str]
    citations: Dict[str, List[str]]


class RedditScraper:
    """Reddit scraper using public APIs and web scraping"""

    def __init__(self):
        self.session = requests.Session()

        # Set up retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set headers to mimic a browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

    def extract_username(self, profile_url: str) -> str:
        """Extract username from Reddit profile URL"""
        parsed = urlparse(profile_url)
        path_parts = parsed.path.strip('/').split('/')

        if len(path_parts) >= 2 and path_parts[0] == 'user':
            return path_parts[1]
        elif len(path_parts) >= 2 and path_parts[0] == 'u':
            return path_parts[1]
        else:
            raise ValueError(f"Invalid Reddit profile URL: {profile_url}")

    def get_user_content(self, username: str, limit: int = 100) -> List[RedditContent]:
        """
        Fetch user's posts and comments using Reddit's JSON API

        Note: This method uses Reddit's public JSON endpoints which may have
        limitations. For production use, consider using Reddit's official API.
        """
        content = []

        # Get user's posts
        posts_url = f"https://www.reddit.com/user/{username}/submitted.json"
        posts_content = self._fetch_json_content(posts_url, limit)

        for post in posts_content:
            content.append(RedditContent(
                id=post['id'],
                title=post.get('title', ''),
                body=post.get('selftext', ''),
                subreddit=post.get('subreddit', ''),
                score=post.get('score', 0),
                created_utc=post.get('created_utc', 0),
                permalink=f"https://www.reddit.com{post.get('permalink', '')}",
                content_type='post'
            ))

        # Get user's comments
        comments_url = f"https://www.reddit.com/user/{username}/comments.json"
        comments_content = self._fetch_json_content(comments_url, limit)

        for comment in comments_content:
            content.append(RedditContent(
                id=comment['id'],
                title='',
                body=comment.get('body', ''),
                subreddit=comment.get('subreddit', ''),
                score=comment.get('score', 0),
                created_utc=comment.get('created_utc', 0),
                permalink=f"https://www.reddit.com{comment.get('permalink', '')}",
                content_type='comment'
            ))

        return content

    def _fetch_json_content(self, url: str, limit: int) -> List[Dict]:
        params = {'limit': limit}
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return [child['data'] for child in data['data']['children']]



    def analyze_activity_patterns(self, content: List[RedditContent]) -> Dict[str, str]:
        """Analyze user's activity patterns"""
        if not content:
            return {}

        # Analyze posting times
        hours = []
        days = []
        subreddits = []

        for item in content:
            if item.created_utc:
                dt = datetime.fromtimestamp(item.created_utc)
                hours.append(dt.hour)
                days.append(dt.strftime('%A'))
                subreddits.append(item.subreddit)

        # Find most common patterns
        most_active_hour = max(set(hours), key=hours.count) if hours else "Unknown"
        most_active_day = max(set(days), key=days.count) if days else "Unknown"
        favorite_subreddits = sorted(set(subreddits), key=subreddits.count, reverse=True)[:5]

        return {
            'most_active_hour': f"{most_active_hour}:00",
            'most_active_day': most_active_day,
            'favorite_subreddits': ', '.join(favorite_subreddits),
            'total_posts': str(len([c for c in content if c.content_type == 'post'])),
            'total_comments': str(len([c for c in content if c.content_type == 'comment']))
        }


class GeminiPersonaGenerator:
    """Generate user persona using Google Gemini"""

    def __init__(self, api_key: str):
        """Initialize Gemini with API key"""
        if not api_key:
            raise ValueError("Google Gemini API key is required. Get one at https://makersuite.google.com/app/apikey")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def generate_persona(self, username: str, content: List[RedditContent]) -> UserPersona:
        """Generate user persona from Reddit content"""

        # Prepare content for analysis
        text_content = self._prepare_content_for_analysis(content)

        print("Analyzing content with Gemini...")

        # Generate persona components
        demographics = self._analyze_demographics(text_content)
        interests = self._analyze_interests(text_content)
        personality_traits = self._analyze_personality(text_content)
        communication_style = self._analyze_communication_style(text_content)
        goals_motivations = self._analyze_goals_motivations(text_content)
        pain_points = self._analyze_pain_points(text_content)

        # Generate citations
        citations = self._generate_citations(content, {
            'demographics': demographics,
            'interests': interests,
            'personality_traits': personality_traits,
            'communication_style': communication_style,
            'goals_motivations': goals_motivations,
            'pain_points': pain_points
        })

        # Get activity patterns
        activity_patterns = RedditScraper().analyze_activity_patterns(content)

        return UserPersona(
            username=username,
            demographics=demographics,
            interests=interests,
            personality_traits=personality_traits,
            communication_style=communication_style,
            activity_patterns=activity_patterns,
            goals_motivations=goals_motivations,
            pain_points=pain_points,
            citations=citations
        )

    def _prepare_content_for_analysis(self, content: List[RedditContent]) -> str:
        """Prepare content for Gemini analysis"""
        combined_text = []

        for item in content:
            text = f"{item.title} {item.body}".strip()
            if text and len(text) > 10:  # Filter out very short content
                combined_text.append(f"[{item.subreddit}] {text}")

        # Limit content to avoid token limits (Gemini has ~32k token limit)
        full_text = '\n'.join(combined_text)
        if len(full_text) > 20000:  # Conservative limit
            return full_text[:20000] + "\n[Content truncated...]"

        return full_text

    def _analyze_demographics(self, content: str) -> Dict[str, str]:
        """Analyze demographics from content"""
        prompt = f"""
        Analyze the following Reddit posts and comments to infer demographic information about the user.
        Look for clues about age, location, profession, education, relationship status, etc.
        Be conservative and only include information that can be reasonably inferred.

        Return your analysis in valid JSON format with these exact keys:
        {{
            "age_range": "age range or Unknown",
            "location": "location or Unknown",
            "profession": "profession or Unknown",
            "education": "education level or Unknown",
            "relationship_status": "relationship status or Unknown",
            "other": "other relevant demographics or Unknown"
        }}

        Content to analyze:
        {content}
        """

        response = self._call_gemini(prompt)
        return self._parse_json_response(response, {
            'age_range': 'Unknown',
            'location': 'Unknown',
            'profession': 'Unknown',
            'education': 'Unknown',
            'relationship_status': 'Unknown',
            'other': 'Unknown'
        })

    def _analyze_interests(self, content: str) -> List[str]:
        """Analyze interests from content"""
        prompt = f"""
        Analyze the following Reddit posts and comments to identify the user's interests and hobbies.
        Look for topics they frequently discuss, subreddits they're active in, activities they mention.

        Return a JSON array of the top 10 most prominent interests as strings.
        Example format: ["Programming", "Video Games", "Cooking", "Travel"]

        Content to analyze:
        {content}
        """

        response = self._call_gemini(prompt)
        return self._parse_json_response(response, [])

    def _analyze_personality(self, content: str) -> List[str]:
        """Analyze personality traits from content"""
        prompt = f"""
        Analyze the following Reddit posts and comments to identify personality traits.
        Consider communication style, emotional expression, opinions, behavior patterns, how they interact with others.

        Return a JSON array of personality traits as strings.
        Use established personality descriptors (e.g., "Analytical", "Empathetic", "Humorous", "Introverted").
        Example format: ["Analytical", "Helpful", "Detail-oriented", "Skeptical"]

        Content to analyze:
        {content}
        """

        response = self._call_gemini(prompt)
        return self._parse_json_response(response, [])

    def _analyze_communication_style(self, content: str) -> str:
        """Analyze communication style from content"""
        prompt = f"""
        Analyze the following Reddit posts and comments to describe the user's communication style.
        Consider tone, formality level, use of humor, how they structure arguments, vocabulary choices.

        Return a brief description (2-3 sentences) describing their communication style.
        Do not format as JSON - just return the description as plain text.

        Content to analyze:
        {content}
        """

        response = self._call_gemini(prompt)
        return response.strip() if response else "Unknown communication style"

    def _analyze_goals_motivations(self, content: str) -> List[str]:
        """Analyze goals and motivations from content"""
        prompt = f"""
        Analyze the following Reddit posts and comments to identify the user's goals and motivations.
        Look for what drives them, what they're trying to achieve, what's important to them, aspirations they mention.

        Return a JSON array of goals and motivations as strings.
        Example format: ["Career advancement", "Learning new skills", "Financial independence"]

        Content to analyze:
        {content}
        """

        response = self._call_gemini(prompt)
        return self._parse_json_response(response, [])

    def _analyze_pain_points(self, content: str) -> List[str]:
        """Analyze pain points and frustrations from content"""
        prompt = f"""
        Analyze the following Reddit posts and comments to identify the user's pain points and frustrations.
        Look for complaints, problems they're trying to solve, challenges they face, things that annoy them.

        Return a JSON array of pain points as strings.
        Example format: ["Work-life balance", "Time management", "Technical difficulties"]

        Content to analyze:
        {content}
        """

        response = self._call_gemini(prompt)
        return self._parse_json_response(response, [])

    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API with rate limiting and error handling"""
        try:
            # Add small delay to avoid rate limiting
            time.sleep(0.5)

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            print(f"Error calling Gemini: {e}")
            if "quota" in str(e).lower():
                print("API quota exceeded. Check your Gemini API usage at https://makersuite.google.com/")
            return "{}"

    def _parse_json_response(self, response: str, default_value):
        clean_response = response.strip()
        if clean_response.startswith('```json'):
            clean_response = clean_response[7:-3]
        elif clean_response.startswith('```'):
            clean_response = clean_response[3:-3]

        json_start = clean_response.find('{')
        json_end = clean_response.rfind('}') + 1

        if json_start != -1 and json_end != -1:
            return json.loads(clean_response[json_start:json_end])

        array_start = clean_response.find('[')
        array_end = clean_response.rfind(']') + 1

        if array_start != -1 and array_end != -1:
            return json.loads(clean_response[array_start:array_end])

        return default_value

    def _generate_citations(self, content: List[RedditContent], persona_data: Dict) -> Dict[str, List[str]]:
        """Generate citations for persona characteristics"""
        citations = {}

        for category, data in persona_data.items():
            citations[category] = []

            # Simple keyword matching for citations
            for item in content[:20]:  # Limit to first 20 items
                text = f"{item.title} {item.body}".lower()
                if len(text) > 20:  # Only cite substantial content
                    citations[category].append(f"{item.permalink} - {text[:100]}...")

        return citations


def save_persona_to_file(persona: UserPersona, filename: str):
    """Save user persona to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"USER PERSONA: {persona.username}\n")
        f.write("=" * 50 + "\n\n")

        f.write("DEMOGRAPHICS:\n")
        f.write("-" * 20 + "\n")
        for key, value in persona.demographics.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")

        f.write("INTERESTS:\n")
        f.write("-" * 20 + "\n")
        for interest in persona.interests:
            f.write(f"• {interest}\n")
        f.write("\n")

        f.write("PERSONALITY TRAITS:\n")
        f.write("-" * 20 + "\n")
        for trait in persona.personality_traits:
            f.write(f"• {trait}\n")
        f.write("\n")

        f.write("COMMUNICATION STYLE:\n")
        f.write("-" * 20 + "\n")
        f.write(f"{persona.communication_style}\n\n")

        f.write("ACTIVITY PATTERNS:\n")
        f.write("-" * 20 + "\n")
        for key, value in persona.activity_patterns.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")

        f.write("GOALS & MOTIVATIONS:\n")
        f.write("-" * 20 + "\n")
        for goal in persona.goals_motivations:
            f.write(f"• {goal}\n")
        f.write("\n")

        f.write("PAIN POINTS:\n")
        f.write("-" * 20 + "\n")
        for pain_point in persona.pain_points:
            f.write(f"• {pain_point}\n")
        f.write("\n")

        f.write("CITATIONS:\n")
        f.write("-" * 20 + "\n")
        for category, citations in persona.citations.items():
            f.write(f"\n{category.upper()}:\n")
            for i, citation in enumerate(citations[:5], 1):  # Limit to 5 citations per category
                f.write(f"{i}. {citation}\n")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate user persona from Reddit profile using Google Gemini')
    parser.add_argument('profile_url', help='Reddit user profile URL')
    parser.add_argument('--api-key', required=True, help='Google Gemini API key')
    parser.add_argument('--output', '-o', default=None, help='Output filename (default: username_persona.txt)')
    parser.add_argument('--limit', '-l', type=int, default=100, help='Limit number of posts/comments to analyze (default: 100)')

    args = parser.parse_args()

    try:
        # Initialize components
        scraper = RedditScraper()
        generator = GeminiPersonaGenerator(args.api_key)

        # Extract username
        username = scraper.extract_username(args.profile_url)
        print(f"Analyzing Reddit user: {username}")

        # Scrape content
        print("Fetching user content...")
        content = scraper.get_user_content(username, args.limit)

        if not content:
            print("No content found for user. Please check the username and try again.")
            return 1

        print(f"Found {len(content)} posts and comments")

        # Generate persona
        print("Generating persona with Google Gemini...")
        persona = generator.generate_persona(username, content)

        # Save to file
        output_file = args.output or f"{username}_persona.txt"
        save_persona_to_file(persona, output_file)

        print(f"Persona saved to {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
