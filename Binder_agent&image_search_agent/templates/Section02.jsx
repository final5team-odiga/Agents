import React from "react";
import styled from "styled-components";

const StyledSection02 = styled.div`
  align-self: stretch;
  height: 800px;
  max-width: 1000px;
  padding: 135px 48px 70px 48px;
  display: inline-flex;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 40px;
`;

const StyledMainImage = styled.img`
  width: 473px;
  height: 595px;
`;

const StyledTextBlock = styled.div`
  flex: 1 1 0;
  height: 595px;
`;

const StyledHeadline = styled.span`
  color: black;
  font-size: 16px;
  font-family: Spectral;
  font-weight: 700;
  line-height: 24px;
  word-wrap: break-word;
`;

const StyledBodyText = styled.span`
  color: black;
  font-size: 16px;
  font-family: Spectral;
  font-weight: 400;
  line-height: 24px;
  word-wrap: break-word;
`;

const StyledFrame2 = styled.div`
  width: 182px;
  display: inline-flex;
  flex-direction: column;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 40px;
`;

const StyledSubImage = styled.img`
  align-self: stretch;
  height: 228px;
`;

export const Section02 = ({ title, body, imageUrl, subImageUrls }) => {
  return (
    <StyledSection02>
      <StyledMainImage src={imageUrl} alt="Main view" />
      <StyledTextBlock>
        <StyledHeadline>{title}</StyledHeadline>

        <StyledBodyText>
          {body.split("\n").map((line, i) => (
            <React.Fragment key={i}>
              {line}
              <br />
            </React.Fragment>
          ))}
        </StyledBodyText>
      </StyledTextBlock>

      <StyledFrame2>
        {subImageUrls.map((url, i) => (
          <StyledSubImage key={i} src={url} alt={`Sub view ${i + 1}`} />
        ))}
      </StyledFrame2>
    </StyledSection02>
  );
};
