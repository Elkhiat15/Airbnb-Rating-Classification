import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from typing import Optional, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 8


def plot_target_distribution(df: pd.DataFrame, 
                             target_col: str = "review_scores_rating",
                             binned_col: Optional[str] = "rating_category",
                             figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    """Plot distribution of continuous and binned target variable."""

    if binned_col and binned_col in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Continuous distribution
        axes[0].hist(df[target_col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].axvline(df[target_col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[target_col].mean():.2f}')
        axes[0].axvline(df[target_col].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[target_col].median():.2f}')
        axes[0].set_xlabel('Review Score Rating', fontsize=9)
        axes[0].set_ylabel('Frequency', fontsize=9)
        axes[0].set_title('Distribution of Review Scores (Continuous)', fontsize=11)
        axes[0].tick_params(axis='both', labelsize=8)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Binned distribution
        counts = df[binned_col].value_counts().sort_index()
        bars = axes[1].bar(range(len(counts)), counts.values, edgecolor='black', alpha=0.7, color='coral')
        axes[1].set_xticks(range(len(counts)))
        axes[1].set_xticklabels(counts.index, rotation=45, ha='right')
        axes[1].set_xlabel('Rating Category', fontsize=9)
        axes[1].set_ylabel('Count', fontsize=9)
        axes[1].set_title('Distribution of Rating Categories (Binned)', fontsize=11)
        axes[1].tick_params(axis='both', labelsize=8)
        axes[1].grid(alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts.values)):
            pct = (count / len(df)) * 100
            axes[1].text(i, count, f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=7)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.hist(df[target_col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(df[target_col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[target_col].mean():.2f}')
        ax.axvline(df[target_col].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[target_col].median():.2f}')
        ax.set_xlabel('Review Score Rating', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title('Distribution of Review Scores', fontsize=11)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_price_by_city(df: pd.DataFrame, 
                       price_col: str = "price",
                       city_col: str = "city",
                       plot_type: str = "box",
                       figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """Plot price distribution across cities using box or violin plots"""

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Sort cities by median price
    city_order = df.groupby(city_col)[price_col].median().sort_values(ascending=False).index.tolist()
    
    # Plot 1: Box/Violin plot
    if plot_type == "violin":
        sns.violinplot(data=df, x=city_col, y=price_col, order=city_order, ax=axes[0], palette="Set2")
    else:
        sns.boxplot(data=df, x=city_col, y=price_col, order=city_order, ax=axes[0], palette="Set2")
    
    axes[0].set_xlabel('City')
    axes[0].set_ylabel('Price (USD)')
    axes[0].set_title(f'Price Distribution by City ({plot_type.capitalize()} Plot)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(alpha=0.3, axis='y')
    
    # Plot 2: Mean and median comparison
    city_stats = df.groupby(city_col)[price_col].agg(['mean', 'median']).sort_values('median', ascending=False)
    x = np.arange(len(city_stats))
    width = 0.35
    
    axes[1].bar(x - width/2, city_stats['mean'], width, label='Mean', alpha=0.8, color='skyblue', edgecolor='black')
    axes[1].bar(x + width/2, city_stats['median'], width, label='Median', alpha=0.8, color='coral', edgecolor='black')
    axes[1].set_xlabel('City')
    axes[1].set_ylabel('Price (USD)')
    axes[1].set_title('Mean vs Median Price by City')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(city_stats.index, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, 
                             features: Optional[List[str]] = None,
                             method: str = 'pearson',
                             figsize: Tuple[int, int] = (12, 10),
                             annot: bool = True,
                             threshold: float = 0.7) -> plt.Figure:
    """Plot correlation heatmap for numeric features."""

    # Select numeric columns
    if features is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[features].select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr(method=method)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 7})
    
    ax.set_title(f'Feature Correlation Heatmap ({method.capitalize()})', fontsize=11, pad=12)
    ax.tick_params(axis='both', labelsize=8)
    
    # Highlight strong correlations
    if threshold < 1.0:
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if strong_corrs:
            print(f"\nStrong correlations (|r| >= {threshold}):")
            for feat1, feat2, corr_val in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True):
                print(f"  {feat1} <-> {feat2}: {corr_val:.3f}")
    
    plt.tight_layout()
    return fig


def plot_feature_vs_target(df: pd.DataFrame,
                           feature_col: str,
                           target_col: str = "review_scores_rating",
                           categorical_target: Optional[str] = "rating_category",
                           figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    """Plot relationship between a categorical feature and target variable.
    
    Args:
        df: Input DataFrame
        feature_col: Name of categorical feature column
        target_col: Name of continuous target column
        categorical_target: Name of binned target (optional)
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    if categorical_target and categorical_target in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Box plot for continuous target
        sns.boxplot(data=df, x=feature_col, y=target_col, ax=axes[0], palette="Set3")
        axes[0].set_xlabel(feature_col.replace('_', ' ').title())
        axes[0].set_ylabel('Review Score Rating')
        axes[0].set_title(f'{feature_col.replace("_", " ").title()} vs Rating (Continuous)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(alpha=0.3, axis='y')
        
        # Plot 2: Stacked bar chart for categorical target
        cross_tab = pd.crosstab(df[feature_col], df[categorical_target], normalize='index') * 100
        cross_tab.plot(kind='bar', stacked=True, ax=axes[1], colormap='viridis', edgecolor='black')
        axes[1].set_xlabel(feature_col.replace('_', ' ').title())
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_title(f'{feature_col.replace("_", " ").title()} vs Rating Category')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(title='Rating Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(alpha=0.3, axis='y')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        sns.boxplot(data=df, x=feature_col, y=target_col, ax=ax, palette="Set3")
        ax.set_xlabel(feature_col.replace('_', ' ').title())
        ax.set_ylabel('Review Score Rating')
        ax.set_title(f'{feature_col.replace("_", " ").title()} vs Rating')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_geospatial_scatter(df: pd.DataFrame,
                            lat_col: str = "latitude",
                            lon_col: str = "longitude",
                            color_by: str = "review_scores_rating",
                            city_col: str = "city",
                            figsize: Tuple[int, int] = (14, 10),
                            vmin: Optional[float] = None,
                            vmax: Optional[float] = None) -> plt.Figure:
    """Plot geospatial scatter of listings colored by rating or price."""

    # Remove missing coordinates
    df_geo = df[[lat_col, lon_col, color_by, city_col]].dropna()
    
    # Check if color column is categorical or numeric
    is_categorical = not pd.api.types.is_numeric_dtype(df_geo[color_by])
    
    if is_categorical:
        # Map categorical values to numeric for plotting
        unique_categories = sorted(df_geo[color_by].unique())
        category_map = {cat: idx for idx, cat in enumerate(unique_categories)}
        df_geo = df_geo.copy()
        df_geo['_color_numeric'] = df_geo[color_by].map(category_map)
        color_col_to_plot = '_color_numeric'
        
        # Use discrete colormap for categorical data
        n_categories = len(unique_categories)
        if n_categories <= 4:
            colors = ['red', 'orange', 'yellow', 'green'][:n_categories]
        else:
            colors = plt.cm.tab10(range(n_categories))
        cmap = mcolors.ListedColormap(colors)
        vmin, vmax = 0, n_categories - 1
    else:
        # Numeric data - use original column
        color_col_to_plot = color_by
        cmap = 'RdYlGn'
        
        # Auto-calculate vmin/vmax if not provided (use 1st-99th percentile)
        if vmin is None:
            vmin = df_geo[color_by].quantile(0.01)
        if vmax is None:
            vmax = df_geo[color_by].quantile(0.99)
    
    # Create figure with subplots for each city
    cities = sorted(df_geo[city_col].unique())
    n_cities = len(cities)
    cols = 3
    rows = (n_cities + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_cities > 1 else [axes]
    
    for idx, city in enumerate(cities):
        city_data = df_geo[df_geo[city_col] == city]
        
        scatter = axes[idx].scatter(city_data[lon_col], city_data[lat_col], 
                                   c=city_data[color_col_to_plot], cmap=cmap, 
                                   alpha=0.6, s=10, edgecolors='none',
                                   vmin=vmin, vmax=vmax)
        axes[idx].set_xlabel('Longitude', fontsize=8)
        axes[idx].set_ylabel('Latitude', fontsize=8)
        axes[idx].set_title(f'{city} (n={len(city_data):,})', fontsize=9)
        axes[idx].tick_params(axis='both', labelsize=7)
        axes[idx].grid(alpha=0.3)
        
        # Add colorbar or legend
        if is_categorical:
            # Create custom colorbar with category labels
            cbar = plt.colorbar(scatter, ax=axes[idx], ticks=range(n_categories))
            cbar.ax.set_yticklabels(unique_categories)
            cbar.set_label(color_by.replace('_', ' ').title(), fontsize=8)
            cbar.ax.tick_params(labelsize=7)
        else:
            cbar = plt.colorbar(scatter, ax=axes[idx], label=color_by.replace('_', ' ').title())
            cbar.ax.tick_params(labelsize=7)
    
    # Hide unused subplots
    for idx in range(n_cities, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Geospatial Distribution Colored by {color_by.replace("_", " ").title()}', 
                 fontsize=11, y=1.00)
    plt.tight_layout()
    return fig


def plot_missing_values(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """Plot heatmap of missing values across the dataset."""

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Calculate missing percentages
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    
    # Plot 1: Bar chart of missing percentages
    if len(missing_pct) > 0:
        axes[0].barh(range(len(missing_pct)), missing_pct.values, color='coral', edgecolor='black', alpha=0.7)
        axes[0].set_yticks(range(len(missing_pct)))
        axes[0].set_yticklabels(missing_pct.index)
        axes[0].set_xlabel('Missing Values (%)')
        axes[0].set_title('Missing Value Percentage by Column')
        axes[0].grid(alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, v in enumerate(missing_pct.values):
            axes[0].text(v + 0.5, i, f'{v:.1f}%', va='center')
    else:
        axes[0].text(0.5, 0.5, 'No Missing Values!', ha='center', va='center', 
                    fontsize=16, transform=axes[0].transAxes)
        axes[0].axis('off')
    
    # Plot 2: Heatmap of missing values (sample rows)
    sample_size = min(1000, len(df))
    df_sample = df.sample(sample_size, random_state=42)
    
    # Only show columns with missing values
    cols_with_missing = df_sample.columns[df_sample.isnull().any()].tolist()
    
    if cols_with_missing:
        sns.heatmap(df_sample[cols_with_missing].isnull(), cbar=False, 
                   yticklabels=False, cmap='coolwarm', ax=axes[1])
        axes[1].set_xlabel('Columns')
        axes[1].set_title(f'Missing Value Pattern (Sample of {sample_size} rows)')
    else:
        axes[1].text(0.5, 0.5, 'No Missing Values!', ha='center', va='center',
                    fontsize=16, transform=axes[1].transAxes)
        axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame, 
                               columns: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """Plot distributions of numeric features with histograms and KDE."""

    # Select numeric columns
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in [np.float64, np.int64]]
    
    # Create subplots
    n_cols = len(numeric_cols)
    cols = 4
    rows = (n_cols + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        data = df[col].dropna()
        
        # Histogram with KDE
        axes[idx].hist(data, bins=30, edgecolor='black', alpha=0.6, color='skyblue', density=True)
        
        # Add KDE if enough data
        if len(data) > 10:
            data.plot(kind='density', ax=axes[idx], color='red', linewidth=2)
        
        axes[idx].set_xlabel(col.replace('_', ' ').title())
        axes[idx].set_ylabel('Density')
        axes[idx].set_title(f'{col.replace("_", " ").title()}\n(μ={data.mean():.2f}, σ={data.std():.2f})')
        axes[idx].grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Distribution of Numeric Features', fontsize=14, y=1.00)
    plt.tight_layout()
    return fig


def plot_price_by_room_type(df: pd.DataFrame,
                            price_col: str = "price",
                            room_type_col: str = "room_type",
                            figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """Plot price distribution by room type with multiple views."""

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Violin plot (log scale)
    df_plot = df[[price_col, room_type_col]].copy()
    df_plot['log_price'] = np.log1p(df_plot[price_col])
    
    sns.violinplot(data=df_plot, x=room_type_col, y='log_price', ax=axes[0], palette="Set2")
    axes[0].set_xlabel('Room Type')
    axes[0].set_ylabel('Log(Price + 1)')
    axes[0].set_title('Price Distribution by Room Type (Log Scale)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(alpha=0.3, axis='y')
    
    # Plot 2: Mean and count
    room_stats = df.groupby(room_type_col)[price_col].agg(['mean', 'count'])
    room_stats = room_stats.sort_values('mean', ascending=False)
    
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    x = np.arange(len(room_stats))
    width = 0.4
    
    bars1 = ax2.bar(x - width/2, room_stats['mean'], width, label='Mean Price', 
                    alpha=0.8, color='skyblue', edgecolor='black')
    bars2 = ax2_twin.bar(x + width/2, room_stats['count'], width, label='Count', 
                         alpha=0.8, color='coral', edgecolor='black')
    
    ax2.set_xlabel('Room Type')
    ax2.set_ylabel('Mean Price (USD)', color='skyblue')
    ax2_twin.set_ylabel('Number of Listings', color='coral')
    ax2.set_title('Mean Price and Count by Room Type')
    ax2.set_xticks(x)
    ax2.set_xticklabels(room_stats.index, rotation=45, ha='right')
    ax2.tick_params(axis='y', labelcolor='skyblue')
    ax2_twin.tick_params(axis='y', labelcolor='coral')
    ax2.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, room_stats['mean'])):
        ax2.text(bar.get_x() + bar.get_width()/2, val, f'${val:.0f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_reviews_vs_rating(df: pd.DataFrame,
                           reviews_col: str = "number_of_reviews",
                           rating_col: str = "review_scores_rating",
                           figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """Plot relationship between number of reviews and rating."""

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Remove missing values
    df_plot = df[[reviews_col, rating_col]].dropna()
    
    # Scatter plot
    axes[0].scatter(df_plot[reviews_col], df_plot[rating_col], alpha=0.3, s=10)
    axes[0].set_xlabel('Number of Reviews')
    axes[0].set_ylabel('Review Score Rating')
    axes[0].set_title('Reviews vs Rating (Linear Scale)')
    axes[0].grid(alpha=0.3)
    
    # Calculate correlation
    corr = df_plot[[reviews_col, rating_col]].corr().iloc[0, 1]
    axes[0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=axes[0].transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Scatter plot with log scale for reviews
    df_plot = df_plot[df_plot[reviews_col] > 0]  # Remove zeros for log
    axes[1].scatter(df_plot[reviews_col], df_plot[rating_col], alpha=0.3, s=10)
    axes[1].set_xlabel('Number of Reviews (Log Scale)')
    axes[1].set_ylabel('Review Score Rating')
    axes[1].set_title('Reviews vs Rating (Log Scale)')
    axes[1].set_xscale('log')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_amenity_analysis(df: pd.DataFrame,
                          amenity_count_col: str = "amenity_count",
                          rating_col: str = "review_scores_rating",
                          price_col: str = "price",
                          figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """Plot amenity count analysis with rating and price relationships."""

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Check if amenity_count exists
    if amenity_count_col not in df.columns:
        # Try to parse from amenities column
        if 'amenities' in df.columns:
            df_plot = df.copy()
            df_plot[amenity_count_col] = df_plot['amenities'].apply(
                lambda x: len(eval(x)) if isinstance(x, str) and x.startswith('[') else 0
            )
        else:
            print(f"Warning: {amenity_count_col} not found in DataFrame")
            return fig
    else:
        df_plot = df
    
    # Distribution of amenity count
    axes[0].hist(df_plot[amenity_count_col], bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[0].axvline(df_plot[amenity_count_col].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {df_plot[amenity_count_col].mean():.1f}')
    axes[0].axvline(df_plot[amenity_count_col].median(), color='green', linestyle='--',
                   linewidth=2, label=f'Median: {df_plot[amenity_count_col].median():.1f}')
    axes[0].set_xlabel('Number of Amenities')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Amenity Count')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Amenity count vs rating (binned scatter with trend)
    df_clean = df_plot[[amenity_count_col, rating_col, price_col]].dropna()
    
    # Create bins for amenity count
    df_clean['amenity_bins'] = pd.cut(df_clean[amenity_count_col], bins=10)
    amenity_rating = df_clean.groupby('amenity_bins')[rating_col].mean()
    amenity_price = df_clean.groupby('amenity_bins')[price_col].mean()
    
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    x = range(len(amenity_rating))
    
    ax2.plot(x, amenity_rating.values, marker='o', linewidth=2, 
            color='blue', label='Avg Rating')
    ax2_twin.plot(x, amenity_price.values, marker='s', linewidth=2,
                 color='orange', label='Avg Price')
    
    ax2.set_xlabel('Amenity Count (Binned)')
    ax2.set_ylabel('Average Rating', color='blue')
    ax2_twin.set_ylabel('Average Price (USD)', color='orange')
    ax2.set_title('Amenity Count vs Rating and Price')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{int(interval.left)}-{int(interval.right)}' 
                        for interval in amenity_rating.index], rotation=45, ha='right')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    plt.tight_layout()
    return fig