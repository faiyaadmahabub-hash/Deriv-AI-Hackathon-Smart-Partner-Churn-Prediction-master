"""
Competitor extraction from partner feedback and communications.
Uses keyword matching to identify competitor mentions.
"""
import re
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


class CompetitorExtractor:
    """
    Extract competitor mentions from partner feedback/support tickets.
    Uses keyword matching for competitor identification.
    """

    def __init__(self):
        """Initialize with competitor keywords from config."""
        self.competitor_keywords = Config.COMPETITOR_KEYWORDS
        self.threat_keywords = Config.THREAT_TYPE_KEYWORDS
        self.negative_context_words = [
            'better', 'cheaper', 'faster', 'prefer', 'switch', 
            'moving to', 'considering', 'offers more', 'higher rates',
            'lower fees', 'better support', 'leaving for'
        ]
        self.positive_context_words = [
            'staying', 'prefer deriv', 'better than', 'chose deriv',
            'happy with', 'satisfied'
        ]

    def extract_mentions(self, text: str) -> Dict:
        """
        Extract competitor mentions from text.

        Args:
            text: Feedback text, support ticket, email content

        Returns:
            Dict with competitor mentions and sentiment
        """
        if not text:
            return self._empty_result()

        text_lower = text.lower()

        # Find competitor mentions
        mentions = {}
        for competitor, keywords in self.competitor_keywords.items():
            count = sum(text_lower.count(kw.lower()) for kw in keywords)
            if count > 0:
                mentions[competitor] = count

        # Determine sentiment context
        sentiment = self._analyze_sentiment(text_lower, bool(mentions))

        # Identify threat type
        threat_types = self._identify_threat_types(text_lower)

        # Extract specific quotes mentioning competitors
        quotes = self._extract_relevant_quotes(text, mentions.keys())

        return {
            'competitors_mentioned': mentions,
            'total_mentions': sum(mentions.values()),
            'primary_competitor': max(mentions.items(), key=lambda x: x[1])[0] if mentions else None,
            'sentiment': sentiment,
            'threat_types': threat_types,
            'relevant_quotes': quotes
        }

    def _analyze_sentiment(self, text: str, has_competitor: bool) -> float:
        """
        Analyze sentiment of competitive context.
        
        Returns:
            Sentiment score from -1 (negative/threatening) to +1 (positive/loyal)
        """
        if not has_competitor:
            return 0.0

        negative_score = sum(1 for word in self.negative_context_words if word in text)
        positive_score = sum(1 for word in self.positive_context_words if word in text)

        # Calculate net sentiment
        if negative_score + positive_score == 0:
            return -0.3  # Default slightly negative when competitor mentioned

        net_score = (positive_score - negative_score) / (negative_score + positive_score + 1)
        return round(max(-1, min(1, net_score)), 2)

    def _identify_threat_types(self, text: str) -> List[str]:
        """Identify types of competitive threats mentioned."""
        threats = []
        for threat_type, keywords in self.threat_keywords.items():
            if any(kw in text for kw in keywords):
                threats.append(threat_type)
        return threats

    def _extract_relevant_quotes(
            self,
            text: str,
            competitors: List[str],
            context_chars: int = 100
    ) -> List[str]:
        """Extract text snippets around competitor mentions."""
        quotes = []

        for competitor in competitors:
            for keyword in self.competitor_keywords.get(competitor, []):
                pattern = re.compile(
                    rf'.{{0,{context_chars}}}{re.escape(keyword)}.{{0,{context_chars}}}',
                    re.IGNORECASE
                )
                matches = pattern.findall(text)
                quotes.extend([f"...{m.strip()}..." for m in matches[:2]])

        return quotes[:5]  # Limit to 5 quotes

    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            'competitors_mentioned': {},
            'total_mentions': 0,
            'primary_competitor': None,
            'sentiment': 0.0,
            'threat_types': [],
            'relevant_quotes': []
        }

    def batch_extract(self, texts: List[Dict]) -> List[Dict]:
        """
        Extract competitor mentions from multiple text sources.

        Args:
            texts: List of dicts with 'partner_id' and 'text' keys

        Returns:
            List of extraction results with partner_id
        """
        results = []
        for item in texts:
            extraction = self.extract_mentions(item.get('text', ''))
            extraction['partner_id'] = item.get('partner_id')
            extraction['source'] = item.get('source', 'unknown')
            results.append(extraction)
        return results

    def get_competitor_summary(self, extractions: List[Dict]) -> Dict:
        """
        Summarize competitor mentions across all extractions.

        Args:
            extractions: List of extraction results

        Returns:
            Summary with competitor counts and threat analysis
        """
        competitor_counts = {}
        threat_type_counts = {}
        total_negative = 0
        total_positive = 0

        for ext in extractions:
            for competitor, count in ext.get('competitors_mentioned', {}).items():
                competitor_counts[competitor] = competitor_counts.get(competitor, 0) + count

            for threat in ext.get('threat_types', []):
                threat_type_counts[threat] = threat_type_counts.get(threat, 0) + 1

            sentiment = ext.get('sentiment', 0)
            if sentiment < 0:
                total_negative += 1
            elif sentiment > 0:
                total_positive += 1

        return {
            'competitor_frequency': competitor_counts,
            'top_competitor': max(competitor_counts.items(), key=lambda x: x[1])[0] if competitor_counts else None,
            'threat_type_distribution': threat_type_counts,
            'sentiment_breakdown': {
                'negative': total_negative,
                'neutral': len(extractions) - total_negative - total_positive,
                'positive': total_positive
            },
            'total_partners_with_mentions': sum(1 for e in extractions if e.get('total_mentions', 0) > 0)
        }


if __name__ == "__main__":
    # Test competitor extractor
    extractor = CompetitorExtractor()

    print("Competitor Extractor Test")
    print("=" * 60)

    test_texts = [
        "I've been looking at Exness rates and they seem cheaper than what Deriv offers.",
        "XM Global has better platform features. Considering switching.",
        "Happy with Deriv, no complaints at all.",
        "IC Markets and Pepperstone both offer lower spreads. We need to discuss rates.",
        "General feedback: need faster support response times."
    ]

    for i, text in enumerate(test_texts, 1):
        result = extractor.extract_mentions(text)
        print(f"\nTest {i}: {text[:50]}...")
        print(f"  Competitors: {result['competitors_mentioned']}")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Threat Types: {result['threat_types']}")
