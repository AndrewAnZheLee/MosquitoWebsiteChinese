import statsmodels.api as sm
import pmdarima as pm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import datetime
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import math
import shap
from matplotlib import patches
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties 
# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Path to the CSV file
DATA_FILE = 'data/mosquito_data.csv'

# Global variable to hold mosquito data (initially empty or loaded from a CSV)
mosquito_data = pd.DataFrame()

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    global mosquito_data
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Load the new data from the uploaded CSV file
            new_data = pd.read_csv(file)
            mosquito_data = new_data

            # Optionally save the updated data to a CSV file
            mosquito_data.to_csv('data/mosquito_data.csv', index=False)

            flash('Data uploaded successfully!', 'success')
            return redirect(url_for('admin'))
    return render_template('admin.html')

# Main index route for predictions
@app.route('/', methods=['GET'])
def index():
    # Get predictions and metrics for the entire dataset
    global importance_dict
    global selected_date_str
    global prediction88
    data = pd.read_csv(DATA_FILE)
    data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Manually parse 'date' after reading
    predictions, metrics, importance_dict , shap = make_predictions(data)
    today = datetime.datetime.now().date()
    tomorrow = today + datetime.timedelta(days=1)

    # Date ranges for the past and future
    past_dates = [(today - datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
    future_dates = [(today + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(2, 8)]
    
    # Selected date
    selected_date_str = request.args.get('selected_date', today.strftime('%Y-%m-%d'))
    # Fetch the prediction
    prediction88 = make_prediction_for_date(selected_date_str)
    # Fetch prediction for the selected date
    return render_template('index.html',
                           predictions=round(prediction88,2),
                           metrics=metrics,
                           today=today.strftime('%Y-%m-%d'),
                           tomorrow=tomorrow.strftime('%Y-%m-%d'),
                           past_dates=past_dates,
                           future_dates=future_dates,
                           selected_date=selected_date_str,
                           feature_importances=importance_dict,
                           sarima_order=order, 
                           sarima_seasonal_order=seasonal_order)

# Route to display a plot
@app.route('/plot.png')
def plot_png():
    try:
        data = pd.read_csv(DATA_FILE)
        data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Manually parse 'date'
    except FileNotFoundError:
        flash('No data available to generate the plot.', 'danger')
        return redirect(url_for('index'))

    predictions,metrics, importance_dict, shap  = make_predictions(data)
    fig = create_plot(data, predictions)
    
    # Save plot to BytesIO and return as PNG
    output = io.BytesIO()
    fig.savefig(output, format='png')
    output.seek(0)
    plt.close(fig)
    return send_file(output, mimetype='image/png')
@app.route('/feature_importances.png')
def feature_importances_png():
        # Convert importance_dict back into a DataFrame
        importance_df = pd.DataFrame(importance_dict)
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Create the plot
        fig = plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='blue')
        plt.xlabel('Importance')
        plt.title('Feature Importances from Random Forest')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save plot to BytesIO and return as PNG
        output = io.BytesIO()
        fig.savefig(output, format='png')
        output.seek(0)
        plt.close(fig)
        return send_file(output, mimetype='image/png')
@app.route('/shap_plot.png')
def shap_plot_png():
    # Return SHAP plot as PNG for web display
    data = pd.read_csv(DATA_FILE)
    shap_output = make_predictions(data)[3]  # Assuming make_predictions returns shap_output as the 4th item
    return send_file(shap_output, mimetype='image/png')

@app.route('/mosquito_density_gauge')
def mosquito_density_gauge():
    # Fetch the prediction
    prediction88 = make_prediction_for_date(selected_date_str)
    predicted_value = round(prediction88,1)
    colors = [(0, "green"), (0.5, "yellow"), (1, "purple")]
    cmap = LinearSegmentedColormap.from_list("green_purple", colors)
    chinese_font = FontProperties(fname='SimHei.ttf') # 确保路径指向系统中的中文字体

    if predicted_value <= 0.5:
        color = "green"
        label = "低"
        suggestion = "蚊蟲活躍度不高，安心上學"
    elif 0.5 < predicted_value <= 1:
        color = "yellow"
        label = "中等"
        suggestion = "蚊蟲活躍度中等，建議使用防蚊液"
    elif 1 < predicted_value <= 1.5:
        color = "orange"
        label = "高"
        suggestion = "蚊蟲活躍度高，建議穿長袖衣物並使用防蚊液"
    else:
        color = "purple"
        label = "危險"
        suggestion = "蚊蟲活躍度危險，建議避免外出，保持門窗緊閉"
    
    # 设置图表和轴
    fig, ax = plt.subplots(figsize=(7, 7))
    
    ### 1. 显示日期 ###
    ax.text(1, 2.1, selected_date_str, fontsize=18, ha='center', va='center', color='black', fontproperties=chinese_font)
    
    ### 2. 水平颜色条 ###
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 2, 1.7, 1.9])

    # 在条形图两端添加标签
    ax.text(0, 1.6, "0", ha='center', va='center', fontsize=12, color='black')
    ax.text(2, 1.6, "2", ha='center', va='center', fontsize=12, color='black')

    # 在条形图上添加指针以表示预测值
    ax.plot([predicted_value], [1.9], marker="v", markersize=20, color="black")

    ### 3. 圆形仪表盘 ###
    # 创建外圆，表示蚊虫密度指数
    outer_circle = Circle((1, 0.9), 0.55, color=color, fill=False, lw=10)
    ax.add_patch(outer_circle)

    # 在圆圈中央显示蚊虫密度值
    ax.text(1, 0.95, f"{predicted_value:.1f}", fontsize=50, ha='center', va='center', color='black', fontweight='bold')

    # 在密度值上方显示“蚊虫密度指数”标签，并指定中文字体
    ax.text(1, 1.15, "Mosquito Density Index", fontsize=12, ha='center', va='center', color='black', fontweight='bold')

    # 在密度值下方显示描述性标签（如低，中等，高等），并指定中文字体
    ax.text(1, 0.65, label, fontsize=18, ha='center', va='center', color='black', fontproperties=chinese_font)

    ### 4. 添加建议 ###
    # 根据蚊虫密度值添加建议文本，使用中文字体
    ax.text(1, 0.15, suggestion, ha='center', va='center', fontsize=12, color='black', wrap=True, fontproperties=chinese_font)

    # 设置轴的范围并移除刻度和标签
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.axis('off')

    # 将图像保存到 BytesIO 对象中以返回图像
    output = io.BytesIO()
    fig.savefig(output, format='png', bbox_inches='tight')
    output.seek(0)
    plt.close(fig)

    return send_file(output, mimetype='image/png')

@app.route('/random_forest_schematic.png')
def random_forest_schematic():
    fig, ax = plt.subplots(figsize=(9, 4))
    save_path= os.path.join('static', 'random_forest_schematic.png')
    # Start with input at the top
    ax.text(0.5, 0.9, 'Input Features\n(Temperature, Humidity, etc.)', 
            bbox=dict(facecolor='lightblue', edgecolor='black'), ha='center', va='center')

    # Define positions for the decision trees (placing each tree next to the other horizontally)
    tree_positions = [(0.2, 0.7), (0.4, 0.7), (0.6, 0.7), (0.8, 0.7)]
    
    for pos_x, pos_y in tree_positions:
        draw_tree(ax, pos_x, pos_y)

    # Draw the prediction at the bottom
    ax.text(0.5, 0.2, 'Final Prediction\n(Mosquito Density)', 
            bbox=dict(facecolor='lightyellow', edgecolor='black'), ha='center', va='center')

    # Turn off axis for cleaner appearance
    ax.axis('off')

    # Save schematic to a file
    output = io.BytesIO()
    fig.savefig(output, format='png')
    output.seek(0)
    fig.savefig(save_path, format='png', bbox_inches='tight')
    plt.close(fig)
    return send_file(output, mimetype='image/png')

def draw_tree(ax, pos_x, pos_y):
    """Draws a simple decision tree structure with three levels"""
    # Draw the root node
    root = patches.Circle((pos_x, pos_y), 0.03, facecolor='green', edgecolor='black')
    ax.add_patch(root)

    # First level branches
    ax.annotate('', xy=(pos_x - 0.05, pos_y - 0.15), xytext=(pos_x, pos_y),
                arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('', xy=(pos_x + 0.05, pos_y - 0.15), xytext=(pos_x, pos_y),
                arrowprops=dict(facecolor='black', shrink=0.05))

    # Second level nodes
    left = patches.Circle((pos_x - 0.05, pos_y - 0.15), 0.03, facecolor='blue', edgecolor='black')
    right = patches.Circle((pos_x + 0.05, pos_y - 0.15), 0.03, facecolor='blue', edgecolor='black')
    ax.add_patch(left)
    ax.add_patch(right)

    # Second level branches
    ax.annotate('', xy=(pos_x - 0.1, pos_y - 0.25), xytext=(pos_x - 0.05, pos_y - 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('', xy=(pos_x, pos_y - 0.25), xytext=(pos_x - 0.05, pos_y - 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('', xy=(pos_x, pos_y - 0.25), xytext=(pos_x + 0.05, pos_y - 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('', xy=(pos_x + 0.1, pos_y - 0.25), xytext=(pos_x + 0.05, pos_y - 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05))

    # Third level nodes
    lower_left = patches.Circle((pos_x - 0.1, pos_y - 0.25), 0.03, facecolor='orange', edgecolor='black')
    lower_middle_left = patches.Circle((pos_x, pos_y - 0.25), 0.03, facecolor='orange', edgecolor='black')
    lower_middle_right = patches.Circle((pos_x, pos_y - 0.25), 0.03, facecolor='orange', edgecolor='black')
    lower_right = patches.Circle((pos_x + 0.1, pos_y - 0.25), 0.03, facecolor='orange', edgecolor='black')
    ax.add_patch(lower_left)
    ax.add_patch(lower_middle_left)
    ax.add_patch(lower_middle_right)
    ax.add_patch(lower_right)


def make_predictions(data):
    global mean_error
    global std_error
    global order
    global seasonal_order
    global sarima_forecast
    """Generate predictions using SARIMA and Random Forest models."""
    train_data = data[data['mosquito_density'].notna()]
    predict_data = data[data['mosquito_density'].isna()]

    if train_data.empty:
        flash('No training data available.', 'danger')
        return None, {}

    # Train SARIMA model
    mosquito_density_train = train_data['mosquito_density']
    weather_features_train = train_data[['temp', 'precip', 'solarradiation', 'humidity', 'windspeed']]
    sarima_model_auto = pm.auto_arima(
        mosquito_density_train,
        seasonal=True,
        m=7,  # weekly seasonality
        stepwise=True,  # perform stepwise search to reduce computation time
        suppress_warnings=True,
        trace=True  # prints the progress of the search
    )

    # Extract the best order and seasonal order
    order = sarima_model_auto.order
    seasonal_order = sarima_model_auto.seasonal_order    
    sarima_model = sm.tsa.statespace.SARIMAX(mosquito_density_train,
                                             order=order,
                                             seasonal_order=seasonal_order,
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)
    try:
        sarima_fit = sarima_model.fit(disp=False)
    except Exception as e:
        flash(f'SARIMA model fitting failed: {e}', 'danger')
        return None, {}

    # Make predictions with SARIMA
    prediction_steps = len(data)
    sarima_forecast = sarima_fit.get_prediction(start=0, end=prediction_steps - 1).predicted_mean
    sarima_forecast = [max(0, forecast) for forecast in sarima_forecast]
    sarima_forecast_train = sarima_fit.get_prediction(start=0, end=len(train_data) - 1).predicted_mean
    sarima_forecast_train = [max(0, forecast) for forecast in sarima_forecast_train]
    # Train Random Forest model on SARIMA residuals
    residuals = mosquito_density_train - sarima_forecast_train
    rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
    rf_model.fit(weather_features_train, residuals)
    
    # Predict residuals for all data
    weather_features_all = data[['temp', 'precip', 'solarradiation', 'humidity', 'windspeed']]
    rf_residuals = rf_model.predict(weather_features_all)

    # Combine SARIMA forecast and RF residuals for hybrid prediction
    hybrid_forecast = sarima_forecast + rf_residuals
    hybrid_forecast = [max(0, forecast) for forecast in hybrid_forecast]
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(weather_features_train)

    # Save SHAP summary plot
    shap_fig = plt.figure()
    shap.summary_plot(shap_values, weather_features_train, show=False)
    shap_output = io.BytesIO()
    shap_fig.savefig(shap_output, format='png')
    shap_output.seek(0)
    plt.close(shap_fig)
    # Save predictions
    predictions_df = pd.DataFrame({
        'date': data['date'],
        'predicted_mosquito_density': hybrid_forecast
    })
    predictions_file = 'predicted_mosquito_density.csv'
    predictions_df.to_csv(predictions_file, index=False)
    # Calculate metrics
    mae = mean_absolute_error(mosquito_density_train, sarima_forecast_train + rf_model.predict(weather_features_train))
    rmse = np.sqrt(mean_squared_error(mosquito_density_train, sarima_forecast_train + rf_model.predict(weather_features_train)))
    mean_error = np.mean((-mosquito_density_train + (sarima_forecast_train + rf_model.predict(weather_features_train))) / (sarima_forecast_train + rf_model.predict(weather_features_train)))
    std_error = np.std((-mosquito_density_train + (sarima_forecast_train + rf_model.predict(weather_features_train))) / (sarima_forecast_train + rf_model.predict(weather_features_train)))

    maeS = mean_absolute_error(mosquito_density_train, sarima_forecast_train)
    rmseS = np.sqrt(mean_squared_error(mosquito_density_train, sarima_forecast_train ))

    
    importances = rf_model.feature_importances_
    features = weather_features_train.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    

    # Convert the DataFrame to a dictionary for passing to the template
    importance_dict = importance_df.to_dict(orient='records')
    # Package the metrics
    metrics = {
        'MAE': round(mae,2),
        'MAES':round(maeS,2),
        'RMSE': round(rmse,2),
        'RMSES':round(rmseS,2),
        'Mean Error': round(mean_error*100,2),
        'STD (Percentage)': round(std_error * 100,2),  
    }

    return hybrid_forecast, metrics , importance_dict,shap_output


def make_prediction_for_date(selected_date_str):
    """Fetch prediction for a specific date from the predictions file."""
    # Load the predictions CSV file
    predictions_df = pd.read_csv('predicted_mosquito_density.csv')

    # Ensure the 'date' column is in string format to match with selected_date_str
    predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.strftime('%Y-%m-%d')

    # Fetch the prediction for the selected date
    prediction = predictions_df[predictions_df['date'] == selected_date_str]['predicted_mosquito_density']
    
    # Check if any prediction exists for the selected date
    if prediction.empty:
        return f"No prediction found for the date: {selected_date_str}"
    
    # Return the first prediction found for the selected date
    return prediction.values[0]

    
def create_plot(data, predictions):
    """Create a plot of actual and predicted mosquito density."""
    errors_up = np.abs((mean_error + std_error) * np.array(predictions))
    errors_down = np.abs((std_error - mean_error) * np.array(predictions))
    errors = [errors_down, errors_up]

    data_sorted = data.sort_values('date')
    data_sorted['date'] = pd.to_datetime(data_sorted['date'])
    data_sorted['predicted_density'] = predictions
    date_range = pd.date_range(start='2024-09-05', periods=len(predictions), freq='D')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(date_range, data_sorted['mosquito_density'], label='Actual Mosquito Density', color='red', marker='o')
    ax.errorbar(date_range, data_sorted['predicted_density'], yerr=errors, label='Predicted Mosquito Density', color='purple', marker='x', capsize=3)
    ax.plot(date_range, sarima_forecast, label='Standalone Sarima', color='green', marker='x')
    ax.set_title('Mosquito Density Predictions')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mosquito Density')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)  # Change port to 8080









