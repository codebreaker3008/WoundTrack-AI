/**
 * Wound Report Generator (Node.js)
 * Generates a detailed PDF report for wound analysis
 * 
 * Dependencies:
 *   npm install pdfkit chartjs-node-canvas moment fs-extra path
 */

import PDFDocument from 'pdfkit';
import fs from 'fs-extra';
import path from 'path';
import moment from 'moment';
import { ChartJSNodeCanvas } from 'chartjs-node-canvas';

/**
 * Create a wound report as a PDF
 * @param {Object} woundData - Wound analysis data object
 * @param {string} outputDir - Folder where the report will be saved
 * @returns {string} - Path to generated PDF
 */
export async function generateWoundReport(woundData, outputDir = 'reports') {
  try {
    // Ensure output directory exists
    await fs.ensureDir(outputDir);

    const woundId = woundData.wound_id || 'unknown';
    const reportPath = path.join(
      outputDir,
      `wound_report_${woundId}_${moment().format('YYYYMMDD_HHmmss')}.pdf`
    );

    const doc = new PDFDocument({ margin: 50 });
    const stream = fs.createWriteStream(reportPath);
    doc.pipe(stream);

    // Title
    doc
      .fontSize(22)
      .fillColor('#0066CC')
      .font('Helvetica-Bold')
      .text('Wound Healing Analysis Report', { align: 'center' })
      .moveDown(1.5);

    // --- PATIENT INFO ---
    doc.fontSize(16).fillColor('#0066CC').text('Patient Information');
    doc.moveDown(0.5);
    doc
      .fontSize(11)
      .fillColor('#000')
      .font('Helvetica')
      .text(`Report Date: ${moment().format('MMMM DD, YYYY [at] hh:mm A')}`)
      .text(
        `Analysis Date: ${moment(woundData.timestamp).format(
          'MMMM DD, YYYY [at] hh:mm A'
        )}`
      )
      .text(`Wound ID: ${woundId}`);
    if (woundData.patient_id)
      doc.text(`Patient ID: ${woundData.patient_id}`);
    doc.moveDown(1);

    // --- MEASUREMENTS ---
    doc.fontSize(16).fillColor('#0066CC').text('Wound Measurements');
    doc.moveDown(0.5);

    const m = woundData.measurements || {};
    const tableData = [
      ['Measurement', 'Value', 'Unit'],
      ['Area', `${(m.area_cm2 || 0).toFixed(2)}`, 'cm²'],
      ['Perimeter', `${(m.perimeter_cm || 0).toFixed(2)}`, 'cm'],
      ['Length', `${(m.length_cm || 0).toFixed(2)}`, 'cm'],
      ['Width', `${(m.width_cm || 0).toFixed(2)}`, 'cm']
    ];

    drawTable(doc, tableData, 70, doc.y + 10);
    doc.moveDown(1);
    doc
      .font('Helvetica-Oblique')
      .fontSize(10)
      .fillColor('#333')
      .text(
        'Note: Measurements are estimates. Include a reference object (coin/ruler) in photos for accuracy.'
      )
      .moveDown(1);

    // --- INFECTION RISK ---
    doc.fontSize(16).fillColor('#0066CC').text('Infection Risk Assessment');
    doc.moveDown(0.5);

    const infection = woundData.infection || {};
    const riskLevel = infection.level || 'Unknown';
    const riskProb = (infection.probability || 0) * 100;

    const riskColor =
      riskLevel === 'High'
        ? '#F44336'
        : riskLevel === 'Medium'
        ? '#FF9800'
        : '#4CAF50';

    const interpretation = {
      Low: 'Wound shows minimal signs of infection. Continue current care.',
      Medium: 'Some infection indicators present. Monitor closely for changes.',
      High: 'Significant infection risk detected. Seek medical attention promptly.'
    }[riskLevel] || 'Unable to assess infection risk.';

    const infectionTable = [
      ['Risk Level', 'Probability', 'Assessment'],
      [riskLevel, `${riskProb.toFixed(1)}%`, interpretation]
    ];

    drawTable(doc, infectionTable, 70, doc.y + 10, { headerColor: '#0066CC', firstColColor: riskColor });
    doc.moveDown(1);

    // --- IMAGE ---
    doc.fontSize(16).fillColor('#0066CC').text('Wound Visualization');
    doc.moveDown(0.5);

    const overlayPath = woundData.files?.overlay?.replace('/results/', 'results/');
    if (overlayPath && (await fs.pathExists(overlayPath))) {
      doc.image(overlayPath, { width: 350, height: 250, align: 'center' });
      doc.moveDown(0.3);
      doc
        .font('Helvetica-Oblique')
        .fontSize(10)
        .fillColor('#333')
        .text('Wound segmentation overlay (red area indicates detected wound)');
    } else {
      doc.fontSize(11).fillColor('#333').text('Image not available.');
    }
    doc.moveDown(1);

    // --- CHART (OPTIONAL) ---
    const chartPath = await generateHealingChart(woundData, outputDir);
    if (chartPath) {
      doc.image(chartPath, { width: 400, height: 250, align: 'center' });
      doc.moveDown(0.3);
      doc
        .font('Helvetica-Oblique')
        .fontSize(10)
        .text('Healing progression chart (AI estimation)');
    }
    doc.moveDown(1);

    // --- RECOMMENDATIONS ---
    doc.fontSize(16).fillColor('#0066CC').text('Care Recommendations');
    doc.moveDown(0.5);

    const recs = [
      '• Keep wound clean and dry',
      '• Change dressings as recommended by healthcare provider',
      '• Monitor for signs of infection (redness, swelling, discharge)',
      '• Take photos regularly to track healing progress'
    ];
    if (riskLevel === 'High') {
      recs.unshift('• IMPORTANT: Seek professional medical attention promptly');
      recs.push('• Watch for fever or increased pain');
    } else if (riskLevel === 'Medium') {
      recs.push('• Schedule follow-up if symptoms worsen');
    }

    doc.font('Helvetica').fontSize(11).fillColor('#000');
    recs.forEach(r => doc.text(r));
    doc.moveDown(1.5);

    // --- DISCLAIMER ---
    doc.addPage();
    doc.fontSize(16).fillColor('#0066CC').text('Medical Disclaimer');
    doc.moveDown(0.5);
    doc
      .font('Helvetica')
      .fontSize(11)
      .fillColor('#000')
      .text(
        `IMPORTANT: This report is generated by an AI-assisted wound analysis system for informational and educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.\n\nAlways seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.\n\nReport generated on: ${moment().format('MMMM DD, YYYY [at] hh:mm A')}`
      );

    doc.end();

    await new Promise(resolve => stream.on('finish', resolve));

    console.log(`✅ Report generated: ${reportPath}`);
    return reportPath;
  } catch (err) {
    console.error('❌ Error generating report:', err);
    return null;
  }
}

/**
 * Draws a simple table in the PDF
 */
function drawTable(doc, data, startX, startY, opts = {}) {
  const rowHeight = 22;
  const colWidths = [150, 150, 200];
  const [header, ...rows] = data;

  let y = startY;
  const headerColor = opts.headerColor || '#0066CC';

  // Header
  doc
    .rect(startX, y, colWidths.reduce((a, b) => a + b, 0), rowHeight)
    .fill(headerColor);
  doc.fillColor('white').font('Helvetica-Bold').fontSize(11);
  let x = startX;
  header.forEach((text, i) => {
    doc.text(text, x + 8, y + 6, { width: colWidths[i] - 10 });
    x += colWidths[i];
  });

  // Rows
  y += rowHeight;
  rows.forEach(row => {
    x = startX;
    row.forEach((text, i) => {
      const bg = i === 0 && opts.firstColColor ? opts.firstColColor : i % 2 === 0 ? '#F8F8F8' : '#FFFFFF';
      doc.rect(x, y, colWidths[i], rowHeight).fill(bg);
      doc.fillColor(i === 0 && opts.firstColColor ? 'white' : '#000');
      doc.font(i === 0 ? 'Helvetica-Bold' : 'Helvetica').fontSize(10);
      doc.text(text, x + 8, y + 6, { width: colWidths[i] - 10 });
      x += colWidths[i];
    });
    y += rowHeight;
  });
}

/**
 * Generates a basic healing progress chart as PNG
 */
async function generateHealingChart(woundData, outputDir) {
  try {
    const chartJSNodeCanvas = new ChartJSNodeCanvas({ width: 500, height: 300 });
    const chartData = woundData.progress || [90, 80, 70, 60, 55, 40];
    const config = {
      type: 'line',
      data: {
        labels: ['Day 1', 'Day 3', 'Day 5', 'Day 7', 'Day 9', 'Day 11'],
        datasets: [
          {
            label: 'Healing (%)',
            data: chartData,
            borderColor: '#0066CC',
            fill: false,
            tension: 0.3
          }
        ]
      },
      options: {
        scales: { y: { beginAtZero: true, max: 100 } },
        plugins: { legend: { display: false } }
      }
    };
    const image = await chartJSNodeCanvas.renderToBuffer(config);
    const chartPath = path.join(outputDir, 'healing_chart.png');
    await fs.writeFile(chartPath, image);
    return chartPath;
  } catch (e) {
    console.error('Error generating chart:', e);
    return null;
  }
}
