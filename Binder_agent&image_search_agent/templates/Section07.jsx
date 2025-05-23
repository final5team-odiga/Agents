import React from "react";
import styled from "styled-components";

const StyledImage = styled.img`
  align-self: stretch;
  height: 514px;
`;

const StyledNilssonText = styled.span`
  color: black;
  font-size: 16px;
  font-family: Poppins;
  font-weight: 400;
  line-height: 22.4px;
  letter-spacing: 0.32px;
  word-wrap: break-word;
`;

const StyledQuote = styled.span`
  color: black;
  font-size: 26px;
  font-family: Spectral;
  font-weight: 400;
  line-height: 28.5px;
  word-wrap: break-word;
`;

const StyledQuoteBody = styled.span`
  color: black;
  font-size: 16px;
  font-family: Spectral;
  font-weight: 400;
  line-height: 24px;
  word-wrap: break-word;
`;

const StyledFrame201 = styled.div`
  width: 386px;
  height: 773px;
  max-width: 1000px;
  position: absolute;
  left: 0px;
  top: 0px;
  display: inline-flex;
  flex-direction: column;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 10px;
`;

const StyledFrame202 = styled.div`
  width: 160px;
  position: absolute;
  left: 802px;
  top: 0px;
  display: inline-flex;
  flex-direction: column;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 20px;
`;

const StyledFrame2 = styled.div`
  width: 1004px;
  height: 696px;
  position: absolute;
  left: 48px;
  top: 77px;
`;

const StyledSection07 = styled.div`
  align-self: stretch;
  height: 800px;
  position: relative;
`;

export const Section07 = ({
  title,
  subtitle,
  body,
  imageUrl,
  subImageUrls,
}) => {
  return (
    <StyledSection07>
      <StyledFrame2>
        <StyledFrame201>
          <StyledImage src={imageUrl} />
          <StyledNilssonText>{subtitle}</StyledNilssonText>
        </StyledFrame201>

        <div
          style={{
            width: 306,
            height: 696,
            position: "absolute",
            left: 417,
            top: 0,
          }}
        >
          <StyledQuote>{title}</StyledQuote>
          <StyledQuoteBody>
            {body.split("\n").map((line, i) => (
              <React.Fragment key={i}>
                {line}
                <br />
              </React.Fragment>
            ))}
          </StyledQuoteBody>
        </div>

        <StyledFrame202>
          {subImageUrls.map((url, i) => (
            <img
              key={i}
              src={url}
              alt={`Sub image ${i + 1}`}
              style={{ width: 160, height: 198 }}
            />
          ))}
        </StyledFrame202>
      </StyledFrame2>
    </StyledSection07>
  );
};
