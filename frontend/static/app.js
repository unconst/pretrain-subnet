document.addEventListener('DOMContentLoaded', function() {
    console.log('Document loaded, starting fetch call...');
    fetch('/static/global_state.json')
    .then(response => {
        console.log('Received response:', response);
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        console.log('Data received:', data);
        const lossData = [];
        const timestamps = [];

        for (const key in data) {
            if (data[key].loss !== null && data[key].timestamp !== null) {
                lossData.push(data[key].loss);
                timestamps.push(data[key].timestamp); // Keep as string for Luxon to parse
            }
        }

        console.log('Processed lossData:', lossData);
        console.log('Processed timestamps:', timestamps);

        const ctx = document.getElementById('lossChart').getContext('2d');
        if (!ctx) {
            console.error('Failed to get the canvas context');
            return;
        }

        const lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timestamps, // These are the timestamp strings
                datasets: [{
                    label: 'Loss',
                    data: lossData,
                    fill: false,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            parser: 'yyyy-MM-dd\'T\'HH:mm:ss', // ISO 8601 format
                            unit: 'minute'
                        },
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Loss: ${context.parsed.y}`;
                            }
                        }
                    }
                }
            }
        });
    })
    .catch(error => {
        console.error('Error fetching the global state:', error);
    });
});
