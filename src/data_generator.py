"""
Data Generator for Netflix Recommendation and Churn Prediction
Generates synthetic viewer data with realistic patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

class NetflixDataGenerator:
    def __init__(self, n_users=100000, n_content=5000, n_days=365):
        """
        Initialize Netflix data generator

        Parameters:
        -----------
        n_users : int
            Number of users (simulating 100K for scalability to 10M+)
        n_content : int
            Number of content items (movies/shows)
        n_days : int
            Number of days of historical data
        """
        self.n_users = n_users
        self.n_content = n_content
        self.n_days = n_days
        self.start_date = datetime.now() - timedelta(days=n_days)

        # Content categories
        self.genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance',
                      'Documentary', 'Thriller', 'Animation', 'Fantasy']
        self.content_types = ['Movie', 'Series', 'Documentary']

    def generate_content_catalog(self):
        """Generate content catalog with metadata"""
        print("Generating content catalog...")

        content_data = []
        for content_id in range(self.n_content):
            content_type = np.random.choice(self.content_types, p=[0.4, 0.5, 0.1])
            primary_genre = np.random.choice(self.genres)

            # Quality score (affects popularity)
            quality_score = np.random.beta(5, 2) * 10  # Skewed toward higher quality

            # Release date
            days_ago = np.random.randint(0, 1825)  # Up to 5 years ago
            release_date = datetime.now() - timedelta(days=days_ago)

            content_data.append({
                'content_id': f'CONTENT_{content_id:05d}',
                'content_type': content_type,
                'primary_genre': primary_genre,
                'quality_score': round(quality_score, 2),
                'release_date': release_date,
                'duration_minutes': np.random.randint(30, 180) if content_type == 'Movie' else np.random.randint(20, 60)
            })

        return pd.DataFrame(content_data)

    def generate_user_profiles(self):
        """Generate user profiles with preferences"""
        print("Generating user profiles...")

        user_data = []
        for user_id in range(self.n_users):
            # User demographics
            age = np.random.randint(18, 70)

            # Subscription info
            subscription_start = self.start_date + timedelta(days=np.random.randint(0, self.n_days))
            subscription_plan = np.random.choice(['Basic', 'Standard', 'Premium'], p=[0.3, 0.5, 0.2])

            # Engagement level
            engagement_level = np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.5, 0.3])

            # Genre preferences (user has affinity for 2-3 genres)
            preferred_genres = np.random.choice(self.genres, size=np.random.randint(2, 4), replace=False)

            user_data.append({
                'user_id': f'USER_{user_id:06d}',
                'age': age,
                'subscription_start': subscription_start,
                'subscription_plan': subscription_plan,
                'engagement_level': engagement_level,
                'preferred_genres': '|'.join(preferred_genres)
            })

        return pd.DataFrame(user_data)

    def generate_viewing_history(self, users_df, content_df):
        """Generate viewing history (interactions)"""
        print("Generating viewing history (this may take a moment)...")

        viewing_data = []

        # Sample users for viewing activity (not all users view every day)
        for day in range(0, self.n_days, 7):  # Weekly sampling for efficiency
            current_date = self.start_date + timedelta(days=day)

            # Active users on this day (30% of users)
            active_users = users_df.sample(frac=0.3, random_state=day)

            for _, user in active_users.iterrows():
                # Number of items watched (based on engagement level)
                if user['engagement_level'] == 'Low':
                    n_watches = np.random.poisson(2)
                elif user['engagement_level'] == 'Medium':
                    n_watches = np.random.poisson(5)
                else:  # High
                    n_watches = np.random.poisson(10)

                n_watches = min(n_watches, 20)  # Cap at 20 per week

                # Select content based on user preferences
                user_genres = user['preferred_genres'].split('|')

                for _ in range(n_watches):
                    # 70% chance to watch preferred genre
                    if np.random.random() < 0.7:
                        available_content = content_df[content_df['primary_genre'].isin(user_genres)]
                    else:
                        available_content = content_df

                    if len(available_content) == 0:
                        continue

                    # Select content (higher quality = higher probability)
                    weights = available_content['quality_score'].values / available_content['quality_score'].sum()
                    selected_content = available_content.sample(n=1, weights=weights).iloc[0]

                    # Watch completion rate (0-100%)
                    completion_rate = np.random.beta(6, 2) * 100

                    # Rating (1-5, correlated with quality and completion)
                    base_rating = selected_content['quality_score'] / 2
                    rating = base_rating + np.random.normal(0, 0.5)
                    rating = np.clip(rating, 1, 5)

                    viewing_data.append({
                        'user_id': user['user_id'],
                        'content_id': selected_content['content_id'],
                        'watch_date': current_date + timedelta(days=np.random.randint(0, 7)),
                        'completion_rate': round(completion_rate, 2),
                        'rating': round(rating, 1),
                        'device': np.random.choice(['TV', 'Mobile', 'Desktop', 'Tablet'])
                    })

        return pd.DataFrame(viewing_data)

    def generate_churn_features(self, users_df, viewing_df):
        """Generate features for churn prediction"""
        print("Generating churn features...")

        churn_data = []

        for _, user in users_df.iterrows():
            user_views = viewing_df[viewing_df['user_id'] == user['user_id']]

            # Calculate engagement metrics
            n_views = len(user_views)
            avg_completion = user_views['completion_rate'].mean() if n_views > 0 else 0
            avg_rating = user_views['rating'].mean() if n_views > 0 else 0

            # Days since last activity
            if n_views > 0:
                last_activity = user_views['watch_date'].max()
                days_since_activity = (datetime.now() - last_activity).days
            else:
                days_since_activity = 365

            # Subscription tenure
            tenure_days = (datetime.now() - user['subscription_start']).days

            # Churn probability factors
            churn_score = 0
            if days_since_activity > 30:
                churn_score += 0.3
            if avg_completion < 50:
                churn_score += 0.2
            if avg_rating < 3:
                churn_score += 0.2
            if n_views < 10:
                churn_score += 0.2
            if tenure_days < 90:
                churn_score += 0.1

            # Determine churn (with some randomness)
            churned = 1 if (churn_score > 0.5 and np.random.random() < churn_score) else 0

            churn_data.append({
                'user_id': user['user_id'],
                'age': user['age'],
                'subscription_plan': user['subscription_plan'],
                'engagement_level': user['engagement_level'],
                'tenure_days': tenure_days,
                'total_views': n_views,
                'avg_completion_rate': round(avg_completion, 2),
                'avg_rating': round(avg_rating, 2),
                'days_since_last_activity': days_since_activity,
                'churned': churned
            })

        return pd.DataFrame(churn_data)

    def generate_and_save(self):
        """Generate all data and save to files"""
        # Generate datasets
        content_df = self.generate_content_catalog()
        users_df = self.generate_user_profiles()
        viewing_df = self.generate_viewing_history(users_df, content_df)
        churn_df = self.generate_churn_features(users_df, viewing_df)

        # Save data
        output_dir = '../data/raw'
        os.makedirs(output_dir, exist_ok=True)

        content_df.to_csv(f'{output_dir}/content_catalog.csv', index=False)
        users_df.to_csv(f'{output_dir}/user_profiles.csv', index=False)
        viewing_df.to_csv(f'{output_dir}/viewing_history.csv', index=False)
        churn_df.to_csv(f'{output_dir}/churn_data.csv', index=False)

        print(f"\n=== Data Generation Complete ===")
        print(f"Users: {len(users_df):,}")
        print(f"Content items: {len(content_df):,}")
        print(f"Viewing records: {len(viewing_df):,}")
        print(f"Churn rate: {churn_df['churned'].mean()*100:.2f}%")
        print(f"\nFiles saved to: {output_dir}/")

        return content_df, users_df, viewing_df, churn_df


if __name__ == "__main__":
    # Generate data representing 100K users (scalable to 10M+)
    generator = NetflixDataGenerator(n_users=100000, n_content=5000, n_days=365)
    content_df, users_df, viewing_df, churn_df = generator.generate_and_save()
