import sqlite3
import json
import os
import matplotlib.pyplot as plt

def extract_graphs():
    db_path = 'database.db'
    output_dir = 'extracted_graphs'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT id_title, result_data FROM movie_analysis")
        rows = cursor.fetchall()
    except Exception as e:
        print(f"Error reading database: {e}")
        return

    processed_titles = set()

    for row in rows:
        title_id = row[0]
        
        # Clean title to identify unique datasets (e.g. "Avengers_Endgame_2019_k2" -> "Avengers Endgame 2019")
        parts = title_id.split('_k')
        base_title = parts[0].replace('_', ' ')
        
        if base_title in processed_titles:
            continue
            
        try:
            data = json.loads(row[1])
            k_results = data.get('optimal_k_results', [])
            
            if not k_results or len(k_results) < 2:
                continue
                
            processed_titles.add(base_title)
            
            # Extract data
            k_values = [res['k'] for res in k_results]
            coherence_scores = [res['score'] for res in k_results]
            perplexity_scores = [res['perplexity'] for res in k_results]
            
            # Clean title for display
            display_title = base_title
            # Clean title for filename (remove weird characters)
            safe_filename = "".join([c for c in base_title if c.isalpha() or c.isdigit() or c==' ']).replace(' ', '_').rstrip()

            # Create separate figure for Coherence
            fig1, ax1 = plt.subplots(figsize=(8, 4.5))
            fig1.patch.set_facecolor('#ffffff')
            ax1.set_facecolor('#ffffff')
            
            # Plot Coherence
            ax1.plot(k_values, coherence_scores, marker='o', linestyle='-', color='#2563eb', linewidth=2.5, markersize=8)
            ax1.set_title('📈 Coherence Score (makin tinggi = baik)', fontsize=20, fontweight='bold', color='#1d4ed8', pad=20)
            ax1.set_xlabel('K (Jumlah Topik)', color='#6b7280', fontsize=12, labelpad=10)
            ax1.set_xticks(k_values)
            ax1.tick_params(axis='x', colors='#374151', labelsize=11)
            
            # Hide Y axis completely
            ax1.yaxis.set_visible(False)
            
            # Clean spines
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            ax1.spines['bottom'].set_color('#d1d5db')
            ax1.spines['bottom'].set_linewidth(1.5)
            
            # Add data labels above points
            for k, val in zip(k_values, coherence_scores):
                ax1.annotate(f"{val:.4f}", xy=(k, val), xytext=(0, 10), 
                            textcoords="offset points", ha='center', color='#1d4ed8', fontsize=10)
                
            plt.tight_layout()
            out_coh = os.path.join(output_dir, f"{safe_filename}_Coherence.png")
            
            # Add rounded border (approximate via figure patch)
            rect = plt.Rectangle((0, 0), 1, 1, fill=False, color="#e5e7eb", linewidth=3, 
                                 zorder=-1, transform=fig1.transFigure, clip_on=False)
            fig1.patches.append(rect)
            
            fig1.savefig(out_coh, dpi=300, bbox_inches='tight', facecolor=fig1.get_facecolor(), pad_inches=0.2)
            plt.close(fig1)

            # Create separate figure for Perplexity
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            fig2.patch.set_facecolor('#ffffff')
            ax2.set_facecolor('#ffffff')
            
            # Plot Perplexity
            ax2.plot(k_values, perplexity_scores, marker='o', linestyle='-', color='#059669', linewidth=2.5, markersize=8)
            ax2.set_title('📉 Perplexity Score (makin rendah = baik)', fontsize=20, fontweight='bold', color='#047857', pad=20)
            ax2.set_xlabel('K (Jumlah Topik)', color='#6b7280', fontsize=12, labelpad=10)
            ax2.set_xticks(k_values)
            ax2.tick_params(axis='x', colors='#374151', labelsize=11)
            
            # Hide Y axis completely
            ax2.yaxis.set_visible(False)
            
            # Clean spines
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['bottom'].set_color('#d1d5db')
            ax2.spines['bottom'].set_linewidth(1.5)
            
            # Add data labels below points
            for k, val in zip(k_values, perplexity_scores):
                ax2.annotate(f"{val:.2f}", xy=(k, val), xytext=(0, -18), 
                            textcoords="offset points", ha='center', color='#059669', fontsize=10)
                
            plt.tight_layout()
            out_perp = os.path.join(output_dir, f"{safe_filename}_Perplexity.png")
            
            # Add rounded border
            rect2 = plt.Rectangle((0, 0), 1, 1, fill=False, color="#e5e7eb", linewidth=3, 
                                 zorder=-1, transform=fig2.transFigure, clip_on=False)
            fig2.patches.append(rect2)
            
            fig2.savefig(out_perp, dpi=300, bbox_inches='tight', facecolor=fig2.get_facecolor(), pad_inches=0.2)
            plt.close(fig2)
            
            print(f"Saved styled graphs for {display_title}")
            
        except Exception as e:
            print(f"Failed to process {title_id}: {e}")

    conn.close()
    print("\nExtraction complete! Check the 'extracted_graphs' folder.")

if __name__ == '__main__':
    extract_graphs()
