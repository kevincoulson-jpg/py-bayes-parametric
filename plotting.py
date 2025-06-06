import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def npz_to_data_dict(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X = np.vstack([data['init_X'], data['iter_X']])
    Y = np.vstack([data['init_Y'], data['iter_Y']])
    input_labels = list(data['input_labels'])
    objective_labels = list(data['objective_labels'])
    data_dict = {}
    for i, label in enumerate(input_labels):
        data_dict[label] = X[:, i]
    for i, label in enumerate(objective_labels):
        data_dict[label] = Y[:, i]
    data_dict['iteration'] = np.arange(X.shape[0])
    return data_dict, input_labels, objective_labels

def plot_subobjectives_vs_iteration(npz_path, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_path)
    x_labels = data['_x_labels']
    y_labels = data['_y_labels']
    N = len(data['iteration'])
    iteration = data['iteration']
    X = np.column_stack([data[label] for label in x_labels])
    
    n_sub = len(y_labels)
    fig = make_subplots(rows=n_sub, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for idx, sub_label in enumerate(y_labels):
        sub_y = data[sub_label]
        hovertexts = []
        for i in range(N):
            txt = f"Iteration: {iteration[i]}<br>"
            txt += "<br>".join([f"{x_labels[j]}: {X[i, j]:.4g}" for j in range(len(x_labels))])
            txt += f"<br>{sub_label}: {sub_y[i]:.4g}"
            hovertexts.append(txt)
        fig.add_trace(go.Scatter(
            x=iteration.tolist(),
            y=sub_y.tolist(),
            mode='markers',
            name=sub_label,
            marker=dict(size=8),
            hoverinfo='text',
            hovertext=hovertexts,
            showlegend=False
        ), row=idx+1, col=1)
        fig.update_yaxes(title_text=sub_label, row=idx+1, col=1)
        if idx < n_sub - 1:
            fig.update_xaxes(showticklabels=False, row=idx+1, col=1)
        else:
            fig.update_xaxes(title_text="Iteration", row=n_sub, col=1)
    fig.update_layout(
        title='Objectives vs. Iteration',
        template='plotly_white',
        width=700,
        height=165*n_sub,  # Reduced height per subplot
        margin=dict(l=60, r=30, t=60, b=30)
    )
    outpath = os.path.join(output_dir, 'subobjectives_vs_iteration.html')
    fig.write_html(outpath)
    print(f"Plot saved to {outpath}")

def plot_inputs_vs_iteration_colored(npz_path, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_path)
    x_labels = data['_x_labels']
    y_labels = data['_y_labels']
    N = len(data['iteration'])
    print(f"N: {N}")
    iteration = data['iteration']
    X = np.column_stack([data[label] for label in x_labels])
    Y = np.column_stack([data[label] for label in y_labels])
    n_inputs = len(x_labels)
    n_objectives = len(y_labels)
    hovertexts = []
    for i in range(N):
        txt = f"Iteration: {iteration[i]}<br>"
        txt += "<br>".join([f"{x_labels[j]}: {X[i, j]:.4g}" for j in range(n_inputs)])
        txt += "<br>" + "<br>".join([f"{y_labels[j]}: {Y[i, j]:.4g}" for j in range(n_objectives)])
        hovertexts.append(txt)
    fig = make_subplots(rows=n_inputs, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    traces = []
    for obj_idx, obj_label in enumerate(y_labels):
        color = Y[:, obj_idx]
        for inp_idx, inp_label in enumerate(x_labels):
            trace = go.Scatter(
                x=iteration.tolist(),
                y=X[:, inp_idx].tolist(),
                mode='markers',
                marker=dict(
                    size=10,
                    color=color,
                    colorscale='Viridis',
                    colorbar=dict(
                        title=obj_label,
                        x=0.5,
                        y=-0.25,  # below the last plot
                        xanchor='center',
                        yanchor='bottom',
                        orientation='h',
                        len=0.7
                    ) if inp_idx == 0 else None,
                    cmin=np.min(color),
                    cmax=np.max(color)
                ),
                name=f"{inp_label} vs {obj_label}",
                hoverinfo='text',
                hovertext=hovertexts,
                visible=(obj_idx == 0),
                showlegend=False
            )
            fig.add_trace(trace, row=inp_idx+1, col=1)
            traces.append((obj_idx, inp_idx))
    buttons = []
    for obj_idx, obj_label in enumerate(y_labels):
        visibility = []
        for o in range(n_objectives):
            for i in range(n_inputs):
                visibility.append(o == obj_idx)
        button = dict(
            label=obj_label,
            method="update",
            args=[{"visible": visibility}]
        )
        buttons.append(button)
    for i, inp_label in enumerate(x_labels):
        fig.update_yaxes(title_text=inp_label, row=i+1, col=1)
        if i < n_inputs - 1:
            fig.update_xaxes(showticklabels=False, row=i+1, col=1)
        else:
            fig.update_xaxes(title_text="Iteration", row=n_inputs, col=1)
    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            buttons=buttons,
            x=1.0,
            y=1.18,
            xanchor='right',
            yanchor='top',
            showactive=True
        )],
        width=900,
        height=350*n_inputs,
        margin=dict(l=60, r=30, t=60, b=60),
        title="Input Variables vs. Iteration (colored by objective)",
        template="plotly_white",
        showlegend=False
    )
    outpath = os.path.join(output_dir, 'inputs_vs_iteration_colored.html')
    fig.write_html(outpath)
    print(f"Plot saved to {outpath}")

def plot_pareto_fronts(npz_path, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_path)
    
    # Extract data and labels
    x_labels = data['_x_labels']
    y_labels = data['_y_labels']
    n_objectives = len(y_labels)
    
    # Get objective and input values from the dictionary
    Y = np.column_stack([data[label] for label in y_labels])
    X = np.column_stack([data[label] for label in x_labels])
    N = Y.shape[0]
    
    # Build custom hovertext for each point
    hovertext = []
    for i in range(N):
        txt = f"<b>Iteration: {i}</b><br>"
        txt += "<b>Objectives:</b><br>"
        txt += "<br>".join([f"{y_labels[j]}: {Y[i, j]:.4g}" for j in range(n_objectives)])
        txt += "<br><b>Inputs:</b><br>"
        txt += "<br>".join([f"{x_labels[j]}: {X[i, j]:.4g}" for j in range(len(x_labels))])
        hovertext.append(txt)
    
    # Create scatterplot matrix (splom)
    fig = go.Figure(data=go.Splom(
        dimensions=[dict(label=label, values=Y[:, i]) for i, label in enumerate(y_labels)],
        text=hovertext,
        marker=dict(
            size=7,
            color=np.arange(N),  # Color by iteration number
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Iteration')
        ),
        hoverinfo='text',
        hovertext=hovertext,
        showupperhalf=False
    ))
    
    fig.update_layout(
        title='Pareto Fronts Between Objectives',
        template='plotly_white',
        width=700,
        height=500,  # Reduced height
        showlegend=False,
        margin=dict(l=60, r=30, t=60, b=60)
    )
    
    outpath = os.path.join(output_dir, 'pareto_fronts.html')
    fig.write_html(outpath)
    print(f"Pareto front plot saved to {outpath}")

def plot_objective_contour(npz_path, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_path)
    
    # Extract data and labels
    y_labels = data['_y_labels']
    n_objectives = len(y_labels)
    
    # Get objective values from the dictionary
    Y = np.column_stack([data[label] for label in y_labels])
    
    # Create the main figure
    fig = go.Figure()
    
    # Add initial contour plot (default to first three objectives)
    fig.add_trace(go.Contour(
        x=Y[:, 1],  # x-axis: objective 2
        y=Y[:, 0],  # y-axis: objective 1
        z=Y[:, 2],  # z-axis: objective 3
        colorscale='Viridis',
        showscale=True,
        name='Contour',
        colorbar=dict(title=y_labels[2])
    ))
    
    # Create dropdown buttons for axis selection
    buttons = []
    for i in range(n_objectives):
        for j in range(n_objectives):
            for k in range(n_objectives):
                if i != j and j != k and i != k:  # Ensure different objectives
                    buttons.append(dict(
                        label=f'X: {y_labels[j]}, Y: {y_labels[i]}, Z: {y_labels[k]}',
                        method='update',
                        args=[
                            {'x': [Y[:, j]], 'y': [Y[:, i]], 'z': [Y[:, k]], 'colorbar.title': [{'text': y_labels[k]}]},
                            {'xaxis.title.text': y_labels[j], 'yaxis.title.text': y_labels[i]}
                        ],
                        execute=True
                    ))
    
    # Update layout with dropdown (move to right)
    fig.update_layout(
        title='Objective Space Contour Plot',
        xaxis_title=y_labels[1],  # Default to second objective
        yaxis_title=y_labels[0],  # Default to first objective
        template='plotly_white',
        width=900,
        height=700,
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1.0,
            'y': 1.1,
            'xanchor': 'right',
            'yanchor': 'top'
        }]
    )
    
    outpath = os.path.join(output_dir, 'objective_contour.html')
    fig.write_html(outpath)
    print(f"Contour plot saved to {outpath}")

def create_dashboard_html(output_dir="plots"):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .plot-section {{ margin-bottom: 20px; }}
            .plot-header {{
                background-color: #f0f0f0;
                padding: 10px;
                cursor: pointer;
                border-radius: 4px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .plot-header:hover {{ background-color: #e0e0e0; }}
            .plot-content {{ 
                display: none;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 0 0 4px 4px;
            }}
            .plot-content.active {{ display: block; }}
            iframe {{ width: 100%; height: 520px; border: none; }}  /* Adjusted height */
            .toggle-icon {{ font-size: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="plot-section">
                <div class="plot-header" onclick="togglePlot('objectives')">
                    <span>Objectives vs Iteration</span>
                    <span class="toggle-icon">▼</span>
                </div>
                <div id="objectives" class="plot-content">
                    <iframe src="subobjectives_vs_iteration.html"></iframe>
                </div>
            </div>
            
            <div class="plot-section">
                <div class="plot-header" onclick="togglePlot('pareto')">
                    <span>Pareto Fronts</span>
                    <span class="toggle-icon">▼</span>
                </div>
                <div id="pareto" class="plot-content">
                    <iframe src="pareto_fronts.html"></iframe>
                </div>
            </div>
        </div>
        
        <script>
            function togglePlot(id) {{
                const content = document.getElementById(id);
                const header = content.previousElementSibling;
                const icon = header.querySelector('.toggle-icon');
                
                if (content.classList.contains('active')) {{
                    content.classList.remove('active');
                    icon.textContent = '▼';
                }} else {{
                    content.classList.add('active');
                    icon.textContent = '▲';
                }}
            }}
            
            // Open both plots by default
            document.addEventListener('DOMContentLoaded', function() {{
                togglePlot('objectives');
                togglePlot('pareto');
            }});
        </script>
    </body>
    </html>
    """
    outpath = os.path.join(output_dir, "dashboard.html")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard saved to {outpath}")

if __name__ == "__main__":
    data_path = r"C:/Users/kevin/Documents/py-bayes-parametric/examples/FreeCADFEA/data/bracket_optimization_results.npz"
    plot_subobjectives_vs_iteration(data_path)
    plot_pareto_fronts(data_path)
    create_dashboard_html("plots") 