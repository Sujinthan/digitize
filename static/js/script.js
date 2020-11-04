function init() {
    canvas = document.getElementById('can');
    ctx = canvas.getContext("2d");
    w = canvas.width;
    h = canvas.height;
    ctx.fillStyle = "white";
    var mousePressed = false;
    ctx.fillRect(0, 0, w, h);
    $('#can').mousedown(function (e) {
        mousePressed = true;
        Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
    });

    $('#can').mousemove(function (e) {
        if (mousePressed) {
            Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
        }
    });

    $('#can').mouseup(function (e) {
        mousePressed = false;
    });
    $('#can').mouseleave(function (e) {
        mousePressed = false;
    });
    // Set up touch events for mobile, etc
    canvas.addEventListener("touchstart", function (e) {
        mousePos = getTouchPos(canvas, e);
        var touch = e.touches[0];
        var mouseEvent = new MouseEvent("mousedown", {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }, false);
    canvas.addEventListener("touchend", function (e) {
        var mouseEvent = new MouseEvent("mouseup", {});
        canvas.dispatchEvent(mouseEvent);
    }, false);
    canvas.addEventListener("touchmove", function (e) {
        var touch = e.touches[0];
        var mouseEvent = new MouseEvent("mousemove", {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }, false);

    // Get the position of a touch relative to the canvas
    function getTouchPos(canvasDom, touchEvent) {
        var rect = canvasDom.getBoundingClientRect();
        return {
            x: touchEvent.touches[0].clientX - rect.left,
            y: touchEvent.touches[0].clientY - rect.top
        };
    }
}

function Draw(x, y, isDown) {
    if (isDown) {
        canvas = document.getElementById('can');
        ctx = canvas.getContext("2d");
        ctx.beginPath();

        ctx.lineCap = "round";
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.strokeStyle = "black";
        ctx.lineWidth = 5;
        ctx.stroke();
        ctx.closePath();
    }
    lastX = x; lastY = y;
}
function erase() {


    ctx.clearRect(0, 0, w, h);
    document.getElementById("canvasimg").style.display = "none";

}
document.addEventListener('DOMContentLoaded', function () {

    $('#loadingImage').hide();
    init()
});

function sendData() {
    $('#result').hide();
    $('#loadingImage').show()
    var scratchCanvas = document.getElementById('can');
    var context = scratchCanvas.getContext('2d');
    var dataURL = scratchCanvas.toDataURL("image/png");

    $.ajax({
        type: "POST",
        url: "/imgToText",
        data: {
            imageBase64: dataURL
        }
    }).done(function (data) {
        document.getElementById("result").innerHTML = "This is letter: " + data
        $('#loadingImage').hide();
        $('#result').show();
    });
}