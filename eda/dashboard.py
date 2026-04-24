import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eda.visualize import (
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_geospatial_scatter,
)

from feature_engineering.selection import categorize_rating

# Page configuration
st.set_page_config(
    page_title="Airbnb Rating Classification Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF5A5F;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #484848;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(data_path: str = "data/processed/cleaned.csv"):
    """Load cleaned data with caching."""
    try:
        df = pd.read_csv(data_path)
        
        # Apply rating categorization with default thresholds (0-3, 3-4.51, 4.51-4.91, 4.91-5)
        df['rating_category'] = df['review_scores_rating'].apply(
            lambda x: categorize_rating(x, thresholds=(3.0, 4.51, 4.91))
        )
        
        # Reorder rating_category to desired order
        category_order = ['Low Rating', 'Medium Rating', 'High Rating', 'Very High Rating']
        # Only include categories that exist in the data
        existing_categories = [cat for cat in category_order if cat in df['rating_category'].unique()]
        df['rating_category'] = pd.Categorical(
            df['rating_category'],
            categories=existing_categories,
            ordered=True
        )
        
        return df
    except FileNotFoundError:
        st.error(f"Data file not found: {data_path}")
        return None


@st.cache_data
def load_featured_data(data_path: str = "data/processed/featured.csv"):
    """Load featured data with caching."""
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        return None


@st.cache_resource
def load_dashboard_model(model_path: str = "models/best_model.pkl"):
    """Load the saved dashboard model."""
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        return None


def build_prediction_features(
    featured_df: pd.DataFrame,
    city: str,
    neighbourhood: str,
    property_type: str,
    room_type: str,
    accommodates: int,
    bedrooms: int,
    bathrooms: int,
    host_response_rate: int,
    price: float,
    amenity_count: int,
    number_of_reviews: int,
) -> pd.DataFrame:
    """Build a single-row feature frame matching the trained model schema."""
    neighbourhood_rows = featured_df[featured_df["neighbourhood"] == neighbourhood]
    if neighbourhood_rows.empty:
        neighbourhood_rows = featured_df[featured_df["city"] == city]

    listing_density = int(len(neighbourhood_rows)) if len(neighbourhood_rows) > 0 else 1

    room_type_prices = featured_df.loc[featured_df["room_type"] == room_type, "price"]
    if room_type_prices.empty:
        room_type_median = float(featured_df["price"].median())
        room_type_std = float(featured_df["price"].std())
    else:
        room_type_median = float(room_type_prices.median())
        room_type_std = float(room_type_prices.std())

    room_type_std = room_type_std if room_type_std and not np.isnan(room_type_std) else 1.0
    price_per_bed = float(price / bedrooms) if bedrooms > 0 else float(price)
    price_relative_to_room_type = float((price - room_type_median) / room_type_std)

    return pd.DataFrame([
        {
            "property_type": property_type,
            "room_type": room_type,
            "accommodates": accommodates,
            "bathrooms": float(bathrooms),
            "city": city,
            "host_response_rate": float(host_response_rate),
            "neighbourhood": neighbourhood,
            "bedrooms": float(bedrooms),
            "amenity_count": amenity_count,
            "price_per_bed": price_per_bed,
            "listing_density": listing_density,
            "price_relative_to_room_type": price_relative_to_room_type,
            "log_price": float(np.log1p(price)),
            "log_number_of_reviews": float(np.log1p(number_of_reviews)),
        }
    ])


def get_prediction_label_map(featured_df: pd.DataFrame, model) -> dict:
    """Map model output classes back to human-readable rating labels."""
    model_classes = list(getattr(model, "classes_", []))
    if not model_classes:
        return {}

    if all(isinstance(label, str) for label in model_classes):
        return {label: label for label in model_classes}

    rating_labels = sorted(featured_df["rating_category"].dropna().unique().tolist())
    return {
        class_value: rating_labels[index]
        for index, class_value in enumerate(model_classes)
        if index < len(rating_labels)
    }


def show_eda_section(df):
    """Display EDA visualizations with interactive controls."""
    
    st.markdown('<p class="sub-header">Exploratory Data Analysis</p>', unsafe_allow_html=True)

    # Key Business Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Listings", f"{len(df):,}")
    with col2:
        avg_rating = df['review_scores_rating'].mean()
        st.metric("Avg Rating", f"{avg_rating:.2f}/5.0")
    with col3:
        avg_price = df['price'].mean()
        st.metric("Avg Price", f"${avg_price:.0f}/night")
    with col4:
        high_rated_pct = (df['review_scores_rating'] >= 4.51).sum() / len(df) * 100
        st.metric("High Rated", f"{high_rated_pct:.1f}%", help="Listings with 4.51+ rating")
    with col5:
        avg_capacity = df['accommodates'].mean()
        st.metric("Avg Capacity", f"{avg_capacity:.1f} guests")
    
    # =============================================================================
    # 1. TARGET DISTRIBUTION
    # =============================================================================
    st.markdown("---")
    st.markdown("#### Rating Distribution Analysis")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        show_binned = st.checkbox("Show binned categories", value=True, key="target_binned")
    
    with col2:
        if show_binned and 'rating_category' in df.columns:
            binned_col = 'rating_category'
        else:
            binned_col = None
    
    fig = plot_target_distribution(
        df,
        target_col='review_scores_rating',
        binned_col=binned_col,
        figsize=(10, 4)
    )
    st.pyplot(fig)
    plt.close()
    
    # =============================================================================
    # 2. FEATURE DISTRIBUTION BY CATEGORY
    # =============================================================================
    st.markdown("---")
    st.markdown("#### Feature Distribution by Category")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        y_axis_options = [
            col for col in ["review_scores_rating", "price", "number_of_reviews"]
            if col in df.columns
        ]
        y_col = st.selectbox(
            "Y-axis (Numeric):",
            y_axis_options,
            index=y_axis_options.index("price") if "price" in y_axis_options else 0,
            key="feature_y"
        )
    
    with col2:
        categorical_cols = ['room_type', 'city']
        categorical_cols = [c for c in categorical_cols if c in df.columns]
        x_col = st.selectbox("X-axis (Category):", categorical_cols, key="feature_x")
    
    with col3:
        plot_type = st.selectbox("Plot Type:", ["box", "violin"], key="feature_plot_type")
    
    # Create custom plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    
    if plot_type == "violin":
        sns.violinplot(data=df, x=x_col, y=y_col, ax=ax, palette="Set2")
    else:
        sns.boxplot(data=df, x=x_col, y=y_col, ax=ax, palette="Set2")
    
    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=9)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=9)
    ax.set_title(f'{y_col.replace("_", " ").title()} by {x_col.replace("_", " ").title()}', 
                fontsize=10, pad=10)
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close()
    
    # =============================================================================
    # 3. CORRELATION ANALYSIS
    # =============================================================================
    st.markdown("---")
    st.markdown("#### Feature Correlation Heatmap")
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox("Correlation Method:", ["pearson", "spearman", "kendall"], key="corr_method")
    with col2:
        show_annot = st.checkbox("Show Values", value=True, key="corr_annot")
    
    # Select features for correlation
    numeric_cols_corr = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect(
        "Select Features (leave empty for all numeric):",
        numeric_cols_corr,
        default=[],
        key="corr_features"
    )
    
    features_to_plot = selected_features if selected_features else None
    
    fig = plot_correlation_heatmap(
        df,
        features=features_to_plot,
        method=method,
        annot=show_annot,
        figsize=(9, 7)
    )
    st.pyplot(fig)
    plt.close()
    
    # =============================================================================
    # 4. GEOGRAPHIC DISTRIBUTION
    # =============================================================================
    st.markdown("---")
    st.markdown("#### Geographic Distribution of Listings")
    
    # Selector for color by option
    color_option = st.selectbox(
        "Color by:",
        ["rating_category", "price"],
        key="geo_color"
    )
    
    # Use selected option for geographic visualization
    if 'latitude' in df.columns and 'longitude' in df.columns:
        fig = plot_geospatial_scatter(
            df,
            lat_col='latitude',
            lon_col='longitude',
            color_by=color_option,
            city_col='city',
            figsize=(10, 7)
        )
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Geographic coordinates not available in this dataset.")


def show_predict_section(featured_df):
    """Display prediction demo interface."""
    
    st.markdown('<p class="sub-header">Rating Prediction Demo</p>', unsafe_allow_html=True)
    
    st.info("Interactive model demo. Enter listing details to predict rating category.")

    model = load_dashboard_model()
    if model is None:
        st.error("Saved model not found at models/best_model.pkl")
        return
    if featured_df is None:
        st.error("Featured data not found at data/processed/featured.csv")
        return

    city_options = sorted(featured_df["city"].dropna().astype(str).unique().tolist())
    
    # Create input form
    with st.form("prediction_form"):
        st.markdown("### Listing Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Location**")
            city = st.selectbox("City:", city_options)
            neighbourhood_options = sorted(
                featured_df.loc[
                    featured_df["city"] == city,
                    "neighbourhood"
                ].dropna().astype(str).unique().tolist()
            )
            neighbourhood = st.selectbox(
                "Neighbourhood:",
                neighbourhood_options if neighbourhood_options else ["Unknown"]
            )
            
            st.markdown("**Property**")
            property_type = st.selectbox("Property Type:", ["Apartment", "House", "Condo", 
                                                           "Townhouse", "Loft"])
            room_type = st.selectbox("Room Type:", ["Entire home/apt", "Private room", "Shared room"])
        
        with col2:
            st.markdown("**Capacity**")
            accommodates = st.number_input("Accommodates:", min_value=1, max_value=16, value=2)
            bedrooms = st.number_input("Bedrooms:", min_value=0, max_value=10, value=1)
            bathrooms = st.number_input("Bathrooms:", min_value=0, max_value=10, value=1)
            
            st.markdown("**Host**")
            host_verified = st.checkbox("Host Identity Verified", value=True)
            host_response_rate = st.slider("Host Response Rate (%):", 0, 100, 95)
        
        with col3:
            st.markdown("**Pricing & Amenities**")
            price = st.number_input("Price per Night ($):", min_value=10, max_value=2800, value=100)
            amenity_count = st.number_input("Number of Amenities:", min_value=0, max_value=100, value=20)
            
            st.markdown("**Reviews**")
            number_of_reviews = st.number_input("Total Reviews:", min_value=0, max_value=500, value=10)
        
        # Submit button
        submitted = st.form_submit_button("Predict Rating", type="primary")
    
    if submitted:
        try:
            features_df = build_prediction_features(
                featured_df=featured_df,
                city=city,
                neighbourhood=neighbourhood,
                property_type=property_type,
                room_type=room_type,
                accommodates=accommodates,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                host_response_rate=host_response_rate,
                price=price,
                amenity_count=amenity_count,
                number_of_reviews=number_of_reviews,
            )

            label_map = get_prediction_label_map(featured_df, model)
            predicted_category_raw = model.predict(features_df)[0]
            predicted_category = label_map.get(predicted_category_raw, str(predicted_category_raw))

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features_df)[0]
                probability_map = {
                    label_map.get(label, str(label)): float(prob)
                    for label, prob in zip(model.classes_, probabilities)
                }
            else:
                probability_map = {str(predicted_category): 1.0}

            st.markdown("---")
            st.markdown("### Prediction Results")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("#### Predicted Rating")
                st.metric("Category", predicted_category)

            with col2:
                st.markdown("#### Class Probabilities")
                for label, prob in sorted(probability_map.items(), key=lambda item: item[1], reverse=True):
                    st.write(f"{label}: {prob * 100:.1f}%")

            with st.expander("Input Features Used"):
                st.json(features_df.iloc[0].to_dict())

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<p class="main-header">Airbnb Rating Classification Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown("**Applied Data Science Project** - Spring 2026")
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select View:",
        ["EDA - Exploratory Analysis", "Predict - Model Demo"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Load cleaned data only
    df = load_data("data/processed/cleaned.csv")
    featured_df = load_featured_data("data/processed/featured.csv")
    
    if df is None:
        st.error("Failed to load data. Please check that data files exist in data/processed/")
        st.stop()
    
    # Main content based on selection
    if "EDA" in page:
        show_eda_section(df)
    else:
        show_predict_section(featured_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>Airbnb Rating Classification Dashboard | Built with Streamlit</p>
        <p>Applied Data Science - Spring 2026</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()